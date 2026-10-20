"""Management uses real services and isolated storage, without Pickle uploads."""

import base64

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.core.exceptions import RepositoryError
from app.main import create_app
from scripts.run_local import load_local_env


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'management.db').as_posix()}",
        artifact_storage_dir=tmp_path / "artifacts",
        admin_api_key="a" * 32, operator_api_key="o" * 32,
        max_upload_size_bytes=4096,
    )
    with TestClient(create_app(settings), client=("127.0.0.1", 50000),
                    headers={"X-API-Key": settings.admin_api_key}) as client:
        yield client


def version_body(artifact):
    return {"version": "v1", "onnx_base64": base64.b64encode(artifact).decode("ascii")}


def test_admin_registration_deployment_prediction_and_stop(client, onnx_artifact):
    assert client.post("/models", json={"name": "demo"}).status_code == 201
    response = client.post("/models/demo/versions", json=version_body(onnx_artifact))
    assert response.status_code == 201
    assert "artifact_location" not in response.json()
    deployment = {"model_name": "demo", "version": "v1", "deployment_id": "d1",
                  "resources": {"cpu": 1.0, "memory_mb": 128}}
    response = client.post("/deployments", json=deployment)
    assert response.status_code == 201
    assert response.json()["status"] == "running"
    assert client.post("/deployments", json=deployment).status_code == 409
    assert client.post("/predict/demo", json={"version": "v1", "inputs": [[1, 2]]}).json()["predictions"] == {"Y": [[3, 5]]}
    assert len(client.get("/models").json()) == 1
    assert len(client.get("/models/demo/versions").json()) == 1
    assert len(client.get("/deployments").json()) == 1
    assert client.post("/deployments/d1/stop").json()["status"] == "stopped"
    assert client.post("/deployments/d1/stop").status_code == 409
    assert client.post("/predict/demo", json={"version": "v1", "inputs": [[1, 2]]}).status_code == 503


def test_management_permissions(client):
    client.headers["X-API-Key"] = "o" * 32
    for path in ("/models", "/deployments"):
        assert client.get(path).status_code == 200
    for path, body in (("/models", {"name": "demo"}), ("/models/demo/versions", {}),
                       ("/deployments", {}), ("/deployments/d1/stop", None)):
        assert client.post(path, json=body).status_code == 403
    del client.headers["X-API-Key"]
    assert client.get("/models").status_code == 401
    assert client.post("/models", json={"name": "demo"}).status_code == 401


def test_duplicates_invalid_metadata_and_artifact_bounds(client, onnx_artifact):
    assert client.post("/models", json={"name": " "}).status_code == 422
    assert client.post("/models", json={"name": "demo"}).status_code == 201
    assert client.post("/models", json={"name": "demo"}).status_code == 409
    assert client.get("/models/missing/versions").status_code == 404
    assert client.post("/models/missing/versions", json=version_body(onnx_artifact)).status_code == 404
    for change in ({"version": " "}, {"input_schema": []}, {"onnx_base64": "!"},
                   {"framework": "sklearn"}, {"artifact_location": "C:/private"}):
        assert client.post("/models/demo/versions", json=version_body(onnx_artifact) | change).status_code == 422
    assert client.post("/models/demo/versions", json=version_body(b"x" * 4097)).status_code == 413
    assert client.post("/models/demo/versions", json=version_body(b"x" * 800000)).status_code == 413
    assert client.post("/models/demo/versions", json=version_body(onnx_artifact)).status_code == 201
    assert client.post("/models/demo/versions", json=version_body(onnx_artifact)).status_code == 409
    assert len(list(client.app.state.container.settings.artifact_storage_dir.glob("*.artifact"))) == 1


def test_registration_failure_cleans_only_new_artifact(client, onnx_artifact, monkeypatch):
    client.post("/models", json={"name": "demo"})
    assert client.post("/models/demo/versions", json=version_body(onnx_artifact)).status_code == 201
    container = client.app.state.container
    before = {path: path.read_bytes() for path in container.settings.artifact_storage_dir.glob("*.artifact")}

    def fail(metadata):
        raise RepositoryError("private persistence detail")

    monkeypatch.setattr(container.model_repository, "save_version", fail)
    response = client.post("/models/demo/versions", json=version_body(onnx_artifact) | {"version": "v2"})
    assert response.status_code == 503
    assert "private" not in response.text
    assert {path: path.read_bytes() for path in container.settings.artifact_storage_dir.glob("*.artifact")} == before
    assert len(client.get("/models/demo/versions").json()) == 1


def test_local_env_optional_literal_and_process_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Track the original values even when the loader introduces absent variables.
    monkeypatch.setenv("APP_NAME", "temporary")
    monkeypatch.setenv("OPERATOR_API_KEY", "temporary")
    monkeypatch.delenv("APP_NAME", raising=False)
    monkeypatch.delenv("OPERATOR_API_KEY")
    load_local_env()
    assert Settings().app_name == "Model Serving Platform"
    monkeypatch.setenv("ADMIN_API_KEY", "p" * 32)
    monkeypatch.delenv("OPERATOR_API_KEY", raising=False)
    for name in ("ENABLE_DEV_SWAGGER", "TRUST_SKLEARN_ARTIFACTS"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        '# comment\nAPP_NAME="Local platform"\nADMIN_API_KEY=' + "f" * 32
        + '\nOPERATOR_API_KEY=' + "o" * 32 + '\n', encoding="utf-8-sig")
    load_local_env()
    settings = Settings()
    assert settings.app_name == "Local platform"
    assert settings.admin_api_key == "p" * 32
    assert settings.operator_api_key == "o" * 32
    assert not settings.enable_dev_swagger and not settings.trust_sklearn_artifacts
    # Restore variables introduced through setdefault, not through monkeypatch.
    monkeypatch.delenv("APP_NAME")
    monkeypatch.delenv("OPERATOR_API_KEY")


def test_local_env_invalid_file_does_not_partially_apply(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("APP_NAME", raising=False)
    (tmp_path / ".env").write_text("APP_NAME=unused\nnot an assignment\n")
    with pytest.raises(ValueError, match="line 2"):
        load_local_env()
    assert Settings().app_name == "Model Serving Platform"
