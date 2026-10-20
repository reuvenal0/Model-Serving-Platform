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
