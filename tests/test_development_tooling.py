"""Persistent development tooling regression tests using isolated test storage."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from sqlalchemy.engine import URL

from app.config import Settings
from app.core.container import ServiceContainer
from app.domain.enums import ModelFramework
from examples.seed_development_model import (
    DESCRIPTION, INPUT_SCHEMA, MODEL_NAME, OUTPUT_SCHEMA, VERSION, database_path, seed,
)
from examples.test_live_api import predictions_match


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url=URL.create("sqlite", database=str(tmp_path / "metadata.db")).render_as_string(),
        artifact_storage_dir=tmp_path / "artifacts", trust_sklearn_artifacts=True,
    )


def test_seed_persists_and_reuses_deployment(settings):
    first = seed(settings)
    assert seed(settings) == first
    assert database_path(settings).is_file()
    with ServiceContainer(settings) as container:
        assert len(container.deployment_manager.list_deployments()) == 1
        assert container.model_registry.download_artifact(MODEL_NAME, VERSION)
        assert container.runtime_manager.get_runtime(MODEL_NAME, VERSION).predict([[4.0]]) == pytest.approx([9.0])
    assert list(settings.artifact_storage_dir.glob("*.artifact"))


def test_stopped_deployment_gets_reusable_replacement(settings):
    first = seed(settings)
    with ServiceContainer(settings) as container:
        container.deployment_manager.undeploy(first)
    second = seed(settings)
    assert second == first + "-2"
    assert seed(settings) == second
    with ServiceContainer(settings) as container:
        assert container.deployment_manager.get_deployment(first).status.value == "stopped"
        assert len(container.deployment_manager.list_deployments()) == 2


def test_mismatched_artifact_is_never_loaded(settings):
    with ServiceContainer(settings) as container:
        container.model_registry.register_model(MODEL_NAME, DESCRIPTION)
        container.model_registry.register_version(
            MODEL_NAME, VERSION, ModelFramework.SKLEARN, b"untrusted arbitrary bytes",
            input_schema=INPUT_SCHEMA, output_schema=OUTPUT_SCHEMA,
        )
    with patch("app.infrastructure.execution.sklearn_model_loader.pickle.loads") as loads:
        with pytest.raises(ValueError, match="refusing to deserialize"):
            seed(settings)
        loads.assert_not_called()
    with ServiceContainer(settings) as container:
        assert container.model_registry.download_artifact(MODEL_NAME, VERSION) == b"untrusted arbitrary bytes"
        assert not container.deployment_manager.list_deployments()


def test_trust_required_before_creating_data(settings):
    with pytest.raises(ValueError, match="TRUST_SKLEARN_ARTIFACTS"):
        seed(replace(settings, trust_sklearn_artifacts=False))
    assert not database_path(settings).exists()


@pytest.mark.parametrize("url", ["sqlite://", "sqlite:///:memory:", "sqlite:///file:test?mode=memory&uri=true"])
def test_memory_database_rejected(settings, url):
    with pytest.raises(ValueError, match="persistent"):
        seed(replace(settings, database_url=url))


def test_live_prediction_validation():
    body = dict(model_name=MODEL_NAME, version=VERSION, predictions=[9.00000000001], latency_ms=0.1)
    assert predictions_match(body, [9.0])
    assert not predictions_match({**body, "predictions": [11.0]}, [9.0])
    assert not predictions_match({**body, "latency_ms": float("nan")}, [9.0])


def test_seed_routing_models_preserves_v1_and_operator_configuration(settings):
    from examples.seed_routing_models import seed as seed_routing

    identifier = seed(settings)
    with ServiceContainer(settings) as container:
        before = container.deployment_repository.get_deployment(identifier)
        artifact = container.model_registry.download_artifact(MODEL_NAME, VERSION)
    second = seed_routing(settings)
    assert seed_routing(settings) == second
    with ServiceContainer(settings) as container:
        after = container.deployment_repository.get_deployment(identifier)
        assert after == before
        assert container.model_registry.download_artifact(MODEL_NAME, VERSION) == artifact
        assert len(container.deployment_manager.list_deployments()) == 2
        assert len(container.routing_repository.history(MODEL_NAME)) == 1
        container.routing_manager.configure_routing(MODEL_NAME, {'v1': 100, 'v2': 0}, 1)
    seed_routing(settings)
    with ServiceContainer(settings) as container:
        assert dict(container.routing_repository.get(MODEL_NAME).config.weights) == {'v1': 100, 'v2': 0}
        assert len(container.routing_repository.history(MODEL_NAME)) == 2


def test_seed_routing_rejects_untrusted_existing_v2(settings):
    from examples.seed_routing_models import seed as seed_routing

    seed(settings)
    with ServiceContainer(settings) as container:
        container.model_registry.register_version(
            MODEL_NAME, 'v2', ModelFramework.SKLEARN, b'untrusted',
            input_schema=INPUT_SCHEMA, output_schema=OUTPUT_SCHEMA,
        )
    with patch('app.infrastructure.execution.sklearn_model_loader.pickle.loads', wraps=__import__('pickle').loads) as loads:
        with pytest.raises(ValueError, match='refusing overwrite or deserialization'):
            seed_routing(settings)
        # A known v1 artifact may be loaded for its targeted readiness check;
        # arbitrary v2 bytes must never reach deserialization.
        assert all(call.args[0] != b'untrusted' for call in loads.call_args_list)


def test_seed_routing_requires_existing_running_v1(settings):
    from examples.seed_routing_models import seed as seed_routing

    identifier = seed(settings)
    with ServiceContainer(settings) as container:
        container.deployment_manager.undeploy(identifier)
        before = container.deployment_repository.get_deployment(identifier)
    with pytest.raises(ValueError, match='existing RUNNING'):
        seed_routing(settings)
    with ServiceContainer(settings) as container:
        assert container.deployment_repository.get_deployment(identifier) == before
        assert len(container.deployment_manager.list_deployments()) == 1


@pytest.mark.parametrize('routing,expected_requests', [(False, 4), (True, 6)])
def test_live_api_optional_routing_preserves_phase5_checks(routing, expected_requests):
    import httpx
    import json
    from examples.test_live_api import verify

    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path == '/health':
            body = {'status': 'ok'}
        elif request.url.path.startswith('/deployments/'):
            body = {'deployment_id': 'demo-api-deployment', 'ready': True, 'status': 'running'}
        elif request.url.path.startswith('/routing/'):
            body = {'model_name': MODEL_NAME, 'revision': 1, 'weights': {'v1': 90, 'v2': 10}}
        else:
            payload = json.loads(request.content)
            version = payload.get('version', 'v2')
            predictions = [(2 if version == 'v1' else 3)*row[0] + 1 for row in payload['inputs']]
            body = {'model_name': MODEL_NAME, 'version': version, 'predictions': predictions, 'latency_ms': 0.1}
        return httpx.Response(200, json=body)

    real_client = httpx.Client
    with patch('examples.test_live_api.httpx.Client',
               side_effect=lambda **kwargs: real_client(**kwargs, transport=httpx.MockTransport(respond))):
        assert verify('http://127.0.0.1:8000', 'demo-api-deployment', routing)
    assert len(requests) == expected_requests
