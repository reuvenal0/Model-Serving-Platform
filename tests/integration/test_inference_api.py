import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
from threading import Event
from unittest.mock import Mock

from tests.api_client import TestClient
import numpy as np
import pytest
from sqlalchemy.engine import URL

from app.api.dependencies import get_deployment_manager, get_inference_service
from app.api.middleware import RequestBodyLimitMiddleware
from app.config import Settings
from app.core.container import ServiceContainer
from app.core import exceptions as errors
from app.domain.enums import DeploymentStatus, ModelFramework
from app.domain.resource_config import ResourceConfig
from app.main import create_app
from app.services.deployment_manager import DeploymentHealth
from app.services.inference_service import InferenceResult


@pytest.fixture
def container(tmp_path):
    settings = Settings(
        database_url=URL.create("sqlite", database=str(tmp_path / "api.db")).render_as_string(),
        artifact_storage_dir=tmp_path / "artifacts", trust_sklearn_artifacts=True,
        max_batch_size=4, max_request_body_bytes=1024,
    )
    with ServiceContainer(settings) as container:
        yield container


@pytest.fixture
def deployed(container, sklearn_artifact):
    container.model_registry.register_model("model")
    container.model_registry.register_version("model", "v1", ModelFramework.SKLEARN, sklearn_artifact)
    container.deployment_manager.deploy("model", "v1", ResourceConfig(1, 128), deployment_id="d-1")
    return container


@pytest.fixture
def client(deployed):
    with TestClient(create_app(container=deployed), raise_server_exceptions=False) as client:
        yield client


def test_real_sklearn_full_flow_and_runtime_reuse(client, deployed):
    first = deployed.runtime_manager.get_runtime("model", "v1")
    response = client.post("/predict/model", json={"version": "v1", "inputs": [[3], [4]]})
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == pytest.approx([7, 9])
    assert (body["model_name"], body["version"]) == ("model", "v1")
    assert body["latency_ms"] >= 0
    assert deployed.runtime_manager.get_runtime("model", "v1") is first
    assert deployed.runtime_manager.cache_info().loads == 1
    deployed.deployment_manager.undeploy("d-1")
    assert client.post("/predict/model", json={"version": "v1", "inputs": [[3]]}).status_code == 503
    assert client.get("/health").json() == {"status": "ok"}


def test_real_onnx_full_flow_and_numpy_json_serialization(container, onnx_artifact):
    container.model_registry.register_model("onnx")
    container.model_registry.register_version("onnx", "v1", ModelFramework.ONNX, onnx_artifact)
    container.deployment_manager.deploy("onnx", "v1", ResourceConfig(1, 128), deployment_id="onnx-deployment")
    with TestClient(create_app(container=container)) as client:
        response = client.post("/predict/onnx", json={"version": "v1", "inputs": {"X": [[1, 2], [3, 4]]}})
        assert response.status_code == 200
        assert response.json()["predictions"] == {"Y": [[3.0, 5.0], [7.0, 9.0]]}
        assert client.get("/deployments/onnx-deployment/health").json()["ready"]
        container.deployment_manager.undeploy("onnx-deployment")
        assert client.post("/predict/onnx", json={"version": "v1", "inputs": [[1, 2]]}).status_code == 503


def test_batch_preserves_sample_order_in_one_runtime_call(client, deployed, monkeypatch):
    runtime = deployed.runtime_manager.get_runtime("model", "v1")
    prediction = Mock(wraps=runtime.predict)
    monkeypatch.setattr(runtime, "predict", prediction)
    response = client.post("/predict/model/batch", json={"version": "v1", "inputs": [[4], [3], [0]]})
    assert response.status_code == 200
    assert response.json()["predictions"] == pytest.approx([9, 7, 1])
    prediction.assert_called_once_with([[4], [3], [0]])


@pytest.mark.parametrize("body", [{}, {"version": "", "inputs": [[3]]}, {"version": 1, "inputs": [[3]]},
                                    {"version": " ", "inputs": [[3]]},
                                    {"version": "v1", "inputs": "bytes"},
                                    {"version": "v1", "inputs": [[3]], "extra": True},
                                    {"version": "v1", "inputs": {"X": None}}])
def test_http_request_validation(client, body):
    response = client.post("/predict/model", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("inputs", [[], [[1]] * 5, [1, 2], [[1, 2]], [["3"]],
                                      [[True]], [[1], [2, 3]], [[[[[1]]]]]])
def test_tensor_shape_values_and_batch_limit(client, deployed, inputs):
    response = client.post("/predict/model", json={"version": "v1", "inputs": inputs})
    assert response.status_code == 422
    # Client mistakes do not cause a deployment to fail.
    assert deployed.deployment_manager.get_deployment("d-1").status is DeploymentStatus.RUNNING


def test_nan_json_is_rejected_without_echoing_raw_input(client):
    response = client.post("/predict/model", content='{"version":"v1","inputs":[[NaN]]}',
                           headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "NaN" not in response.text


def test_invalid_json_is_sanitized(client):
    response = client.post("/predict/model", content='{"private":', headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "private" not in response.text


@pytest.mark.parametrize("path,version", [("missing-model", "v1"), ("model", "missing-version")])
def test_missing_model_and_version(client, path, version):
    response = client.post(f"/predict/{path}", json={"version": version, "inputs": [[3]]})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_registered_but_undeployed_version_is_unavailable(client, deployed, sklearn_artifact):
    deployed.model_registry.register_version("model", "v2", ModelFramework.SKLEARN, sklearn_artifact)
    assert client.post("/predict/model", json={"version": "v2", "inputs": [[3]]}).status_code == 503


def test_failed_deployment_is_unavailable(client, deployed):
    snapshot = deployed.deployment_repository.get_deployment("d-1")
    snapshot.deployment.fail()
    deployed.deployment_repository.update_deployment(snapshot.deployment, expected_revision=snapshot.revision)
    assert client.post("/predict/model", json={"version": "v1", "inputs": [[3]]}).status_code == 503


def test_health_endpoints_distinguish_liveness_from_readiness(client, deployed):
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/deployments/d-1/health").json()
    assert body["ready"] and body["status"] == "running" and body["kind"] == "load"
    assert body["checked_at"].endswith(("Z", "+00:00"))
    deployed.deployment_manager.undeploy("d-1")
    response = client.get("/deployments/d-1/health")
    assert response.status_code == 200 and not response.json()["ready"]
    assert client.get("/deployments/missing/health").status_code == 404
    assert client.get("/health").status_code == 200


def test_load_failure_returns_503_without_leaking_storage_path(client, deployed):
    key = deployed.model_registry.get_version("model", "v1").artifact_location
    deployed.artifact_storage.delete(key)
    deployed.runtime_manager.clear()
    response = client.post("/predict/model", json={"version": "v1", "inputs": [[3]]})
    assert response.status_code == 503
    assert key not in response.text
    assert str(deployed.settings.artifact_storage_dir) not in response.text


@pytest.mark.parametrize("error,status", [
    (errors.PredictionInputError("/private/input"), 422),
    (errors.ModelNotFoundError("/private/model"), 404),
    (errors.ModelVersionNotFoundError("/private/version"), 404),
    (errors.DeploymentNotFoundError("/private/deployment"), 404),
    (errors.DeploymentConflictError("/private/conflict"), 409),
    (errors.InvalidDeploymentStateError("/private/state"), 409),
    (errors.ModelLoadingError("/private/artifact.pkl"), 503),
    (errors.ArtifactNotFoundError("/private/missing"), 503),
    (errors.RepositoryError("/private/db"), 503),
    (errors.ModelContractError("/private/schema"), 503),
    (errors.PredictionSerializationError("/private/output"), 500),
    (errors.PredictionError("/private/prediction"), 500),
    (RuntimeError("/private/unexpected"), 500),
])
def test_exception_http_mappings_and_no_detail_leakage(client, error, status):
    fake = Mock()
    fake.predict.side_effect = error
    client.app.dependency_overrides[get_inference_service] = lambda: fake
    response = client.post("/predict/model", json={"version": "v1", "inputs": [[3]]})
    assert response.status_code == status
    assert "private" not in response.text
    assert "Traceback" not in response.text
    assert set(response.json()["error"]) == {"code", "message"}


def test_dependency_overrides_for_both_services(client):
    fake = Mock()
    fake.predict.return_value = InferenceResult("replacement", "v1", [1], 0.25)
    manager = Mock()
    manager.check_health.return_value = DeploymentHealth("override", DeploymentStatus.STOPPED, False,
                                                         "not_running", datetime.now(timezone.utc))
    client.app.dependency_overrides[get_inference_service] = lambda: fake
    client.app.dependency_overrides[get_deployment_manager] = lambda: manager
    assert client.post("/predict/model", json={"version": "v1", "inputs": [[3]]}).json()["model_name"] == "replacement"
    assert client.get("/deployments/override/health").json()["status"] == "stopped"
    fake.predict.assert_called_once_with("model", "v1", [[3]])


def test_body_limit_checks_content_length_and_actual_streamed_bytes(client):
    response = client.post("/predict/model", content=b"x" * 1025, headers={"content-type": "application/json"})
    assert response.status_code == 413
    streamed = client.post("/predict/model", content=iter([b"x" * 600, b"y" * 600]),
                           headers={"content-type": "application/json"})
    assert streamed.status_code == 413
    huge_header = client.post("/predict/model", content=b"{}", headers={"content-length": "9" * 5000})
    assert huge_header.status_code == 413
    malformed = client.post("/predict/model", content=b"{}", headers={"content-length": "invalid"})
    assert malformed.status_code == 422


def test_asgi_stream_limit_stops_reading_and_never_invokes_router():
    async def scenario():
        downstream = Mock()
        app = RequestBodyLimitMiddleware(downstream, max_body_bytes=10)
        messages = iter([{"type": "http.request", "body": b"123456", "more_body": True},
                         {"type": "http.request", "body": b"789012", "more_body": True}])
        sent = []

        async def receive():
            return next(messages)

        async def send(message):
            sent.append(message)

        await app({"type": "http", "headers": []}, receive, send)
        downstream.assert_not_called()
        assert sent[0]["status"] == 413
        assert json.loads(sent[1]["body"])["error"]["code"] == "request_too_large"

    asyncio.run(scenario())


def test_factory_creates_resources_only_during_lifespan_and_disposes_owned_resources(tmp_path, monkeypatch, sklearn_artifact):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'owned.db'}", artifact_storage_dir=tmp_path / "owned-artifacts",
                        trust_sklearn_artifacts=True)
    created = []
    real = ServiceContainer

    def construct(selected):
        instance = real(selected)
        instance.close = Mock(wraps=instance.close)
        instance.engine.dispose = Mock(wraps=instance.engine.dispose)
        created.append(instance)
        return instance

    monkeypatch.setattr("app.main.ServiceContainer", construct)
    app = create_app(settings)
    assert not created
    assert not (tmp_path / "owned.db").exists()
    with TestClient(app) as client:
        owned = created[0]
        owned.model_registry.register_model("model")
        owned.model_registry.register_version("model", "v1", ModelFramework.SKLEARN, sklearn_artifact)
        owned.deployment_manager.deploy("model", "v1", ResourceConfig(1, 128))
        assert client.get("/health").status_code == 200
        assert owned.runtime_manager.cache_info().current_size == 1
    owned.close.assert_called_once()
    owned.engine.dispose.assert_called_once()
    assert owned.runtime_manager.cache_info().current_size == 0
    assert app.state.container is None


def test_injected_container_remains_caller_owned(deployed):
    close = Mock(wraps=deployed.close)
    deployed.close = close
    with TestClient(create_app(container=deployed)) as client:
        assert client.get("/health").status_code == 200
    close.assert_not_called()
    assert deployed.runtime_manager.cache_info().current_size == 1


def test_stop_during_readiness_prevents_http_admission(client, deployed, monkeypatch):
    entered, release = Event(), Event()
    original = deployed.health_checker.check

    def blocked(runtime, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(runtime, **kwargs)

    monkeypatch.setattr(deployed.health_checker, "check", blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, "/predict/model", json={"version": "v1", "inputs": [[3]]})
        assert entered.wait(5)
        try:
            deployed.deployment_manager.undeploy("d-1")
        finally:
            release.set()
        assert pending.result(timeout=10).status_code == 503


def test_scale_during_readiness_returns_revision_conflict(client, deployed, monkeypatch):
    original = deployed.health_checker.check

    def scaled(runtime, **kwargs):
        deployed.deployment_manager.scale("d-1", 3)
        return original(runtime, **kwargs)

    monkeypatch.setattr(deployed.health_checker, "check", scaled)
    assert client.post("/predict/model", json={"version": "v1", "inputs": [[3]]}).status_code == 409


def test_admitted_http_prediction_may_finish_but_next_request_is_rejected(client, deployed, monkeypatch):
    runtime = deployed.runtime_manager.get_runtime("model", "v1")
    original = runtime.predict
    entered, release = Event(), Event()

    def blocked(inputs):
        entered.set()
        assert release.wait(5)
        return original(inputs)

    monkeypatch.setattr(runtime, "predict", blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(client.post, "/predict/model", json={"version": "v1", "inputs": [[3]]})
        assert entered.wait(5)
        try:
            deployed.deployment_manager.undeploy("d-1")
            assert client.post("/predict/model", json={"version": "v1", "inputs": [[3]]}).status_code == 503
        finally:
            release.set()
        response = pending.result(timeout=10)
    assert response.status_code == 200
    assert response.json()["predictions"] == pytest.approx([7])


def test_onnx_integer_dtype_rejects_fractional_json_numbers(container):
    from onnx import TensorProto, helper

    graph = helper.make_graph([helper.make_node("Identity", ["X"], ["Y"])], "ints",
                              [helper.make_tensor_value_info("X", TensorProto.INT32, [None, 1])],
                              [helper.make_tensor_value_info("Y", TensorProto.INT32, [None, 1])])
    artifact = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8).SerializeToString()
    container.model_registry.register_model("integers")
    container.model_registry.register_version("integers", "v1", ModelFramework.ONNX, artifact)
    container.deployment_manager.deploy("integers", "v1", ResourceConfig(1, 128))
    with TestClient(create_app(container=container)) as client:
        assert client.post("/predict/integers", json={"version": "v1", "inputs": [[1.5]]}).status_code == 422
        assert client.post("/predict/integers", json={"version": "v1", "inputs": [[2]]}).json()["predictions"] == {"Y": [[2]]}


def test_openapi_describes_optional_version_and_structured_errors(client):
    spec = client.get("/openapi.json").json()
    request = spec["components"]["schemas"]["PredictionRequest"]
    assert set(request["required"]) == {"inputs"}
    errors = spec["paths"]["/predict/{model_name}"]["post"]["responses"]
    assert errors["422"]["content"]["application/json"]["schema"]["$ref"].endswith("ErrorResponse")


def test_http_inference_lazily_reloads_after_cache_clear(client, deployed):
    deployed.runtime_manager.clear()
    before = deployed.runtime_manager.cache_info().loads
    response = client.post("/predict/model", json={"version": "v1", "inputs": [[3]]})
    assert response.status_code == 200
    assert response.json()["predictions"] == pytest.approx([7])
    assert deployed.runtime_manager.cache_info().loads == before + 1
