"""Exercise the official demo against real APIs and temporary SQLite/artifacts."""

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.main import create_app
from examples.portfolio_demo.demo import BASELINE, DESCRIPTION, MODEL, predict, run, setup


@pytest.fixture
def demo_api(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'demo.db').as_posix()}",
        artifact_storage_dir=tmp_path / "artifacts",
        admin_api_key="a" * 32, trust_sklearn_artifacts=False,
    )
    with TestClient(create_app(settings), client=("127.0.0.1", 50000),
                    headers={"X-API-Key": settings.admin_api_key}) as client:
        def call(method, path, body=None, *, missing_ok=False):
            response = client.request(method, path, json=body)
            if missing_ok and response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        yield call, settings.artifact_storage_dir


def test_onnx_setup_repeat_routing_monitoring_and_rollback(demo_api):
    call, artifacts = demo_api
    setup(call)
    original_models = call("GET", "/models")
    original_versions = call("GET", f"/models/{MODEL}/versions")
    original_deployments = call("GET", "/deployments")
    original_bytes = {path: path.read_bytes() for path in artifacts.glob("*.artifact")}
    setup(call)
    assert call("GET", "/models") == original_models
    assert call("GET", f"/models/{MODEL}/versions") == original_versions
    assert call("GET", "/deployments") == original_deployments
    assert all(item["status"] == "running" for item in original_deployments)
    assert predict(call, "v1")["predictions"] == {"Y": [[9.0]]}
    assert predict(call, "v2")["predictions"] == {"Y": [[13.0]]}
    run(call)
    setup(call)
    history = call("GET", f"/routing/{MODEL}/history")
    assert [item["weights"] for item in history] == [BASELINE, {"v1": 90, "v2": 10}, {"v1": 50, "v2": 50}, BASELINE]
    assert history[-1]["restored_from"] == history[0]["revision"]
    assert call("GET", f"/metrics/{MODEL}")["successful_requests"] == 31
    assert len(call("GET", f"/inference-logs?model_name={MODEL}&limit=5")["logs"]) == 5
    assert {path: path.read_bytes() for path in artifacts.glob("*.artifact")} == original_bytes
    call("POST", "/deployments/portfolio-onnx-v2/stop")
    with pytest.raises(RuntimeError, match="incompatible or stopped"):
        setup(call)
    assert next(item for item in call("GET", "/deployments")
                if item["id"] == "portfolio-onnx-v2")["status"] == "stopped"


def test_incompatible_model_or_routing_is_not_overwritten(demo_api):
    call, artifacts = demo_api
    call("POST", "/models", {"name": MODEL, "description": "User-created model"})
    with pytest.raises(RuntimeError, match="incompatible metadata"):
        setup(call)
    assert call("GET", "/models")[0]["description"] != DESCRIPTION
    assert call("GET", "/deployments") == []
    assert list(artifacts.glob("*.artifact")) == []
