"""Bridge peers never become management peers through a key or spoofed header."""

from secrets import token_urlsafe

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.main import create_app


@pytest.mark.parametrize(
    "path",
    [
        "/metrics",
        "/runtime/resources",
        "/runtime/admission",
        "/routing/model",
        "/docs",
        "/openapi.json",
    ],
)
def test_docker_bridge_peer_is_denied_even_with_key_and_forwarded_loopback(
    tmp_path, path
):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'metadata.db'}",
        artifact_storage_dir=tmp_path / "artifacts",
        admin_api_key=token_urlsafe(32),
    )
    with TestClient(
        create_app(settings),
        client=("172.18.0.1", 50000),
        headers={"X-API-Key": settings.admin_api_key, "X-Forwarded-For": "127.0.0.1"},
    ) as client:
        assert client.get(path).status_code == 403
        assert client.get("/health").status_code == 200


def test_container_local_loopback_requires_a_key_and_opens_docs_only_when_authorized(
    tmp_path,
):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'metadata.db'}",
        artifact_storage_dir=tmp_path / "artifacts",
        admin_api_key=token_urlsafe(32),
    )
    with TestClient(create_app(settings), client=("127.0.0.1", 50000)) as client:
        assert client.get("/metrics").status_code == 401
        assert client.get("/docs").status_code == 401
        assert (
            client.get(
                "/metrics", headers={"X-API-Key": settings.admin_api_key}
            ).status_code
            == 200
        )
        assert (
            client.get(
                "/docs", headers={"X-API-Key": settings.admin_api_key}
            ).status_code
            == 200
        )
        # Public inference reaches existing services; it does not require a key.
        assert (
            client.post(
                "/predict/missing", json={"version": "v1", "inputs": [[4.0]]}
            ).status_code
            == 404
        )
