"""Portable recovery tests for the same fixture used by actual Docker smoke."""

from dataclasses import replace
from secrets import token_urlsafe

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.main import create_app
from scripts import container_fixture


@pytest.fixture
def volume_settings(tmp_path, monkeypatch):
    database = tmp_path / "db"
    database.mkdir()
    settings = Settings(
        database_url=f"sqlite:///{database / 'smoke.db'}",
        artifact_storage_dir=tmp_path / "artifacts",
        trust_sklearn_artifacts=True,
        admin_api_key=token_urlsafe(32),
    )
    monkeypatch.setattr(container_fixture, "require_isolated_storage", lambda: settings)
    return settings


def test_empty_environment_starts_without_example_models(volume_settings):
    app = create_app(replace(volume_settings, trust_sklearn_artifacts=False))
    with TestClient(app, client=("127.0.0.1", 1234)) as client:
        assert client.get("/health").status_code == 200
        assert app.state.container.model_registry.list_models() == []
        assert app.state.container.deployment_manager.list_deployments() == []
    assert app.state.container is None


def test_two_recreations_preserve_data_and_revalidate_deployments(volume_settings):
    container_fixture.seed()
    original = container_fixture.fingerprint()
    assert len(original["tables"]) == 7
    previous_logs = None
    for _ in range(2):
        app = create_app(volume_settings)
        with TestClient(
            app,
            client=("127.0.0.1", 1234),
            headers={"X-API-Key": volume_settings.admin_api_key},
        ) as client:
            assert app.state.container.runtime_manager.cache_info().loads == 1
            assert (
                client.get("/deployments/smoke-running/health").json()["ready"] is True
            )
            assert (
                client.get("/deployments/smoke-interrupted/health").json()["status"]
                == "failed"
            )
            logs = client.get("/inference-logs").json()["logs"]
            if previous_logs:
                assert {entry["request_id"] for entry in previous_logs} <= {
                    entry["request_id"] for entry in logs
                }
            response = client.post("/predict/container-smoke", json={"inputs": [[4.0]]})
            assert response.status_code == 200 and response.json()[
                "predictions"
            ] == pytest.approx([9.0])
            assert client.get("/routing/container-smoke").json()["weights"] == {
                "v1": 100
            }
            previous_logs = client.get("/inference-logs").json()["logs"]
        assert container_fixture.fingerprint() == original
        assert app.state.container is None


def test_fixture_does_not_overwrite_existing_records(volume_settings):
    container_fixture.seed()
    original = container_fixture.fingerprint()
    with pytest.raises(ValueError, match="empty model metadata"):
        container_fixture.seed()
    assert container_fixture.fingerprint() == original


def test_disabled_trust_after_restart_rejects_inference(volume_settings):
    container_fixture.seed()
    app = create_app(replace(volume_settings, trust_sklearn_artifacts=False))
    with TestClient(
        app,
        client=("127.0.0.1", 1234),
        headers={"X-API-Key": volume_settings.admin_api_key},
    ) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/deployments/smoke-running/health").json()["ready"] is False
        assert (
            client.post(
                "/predict/container-smoke", json={"version": "v1", "inputs": [[4.0]]}
            ).status_code
            == 503
        )
