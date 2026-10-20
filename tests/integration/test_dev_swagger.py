"""Browser documentation is not an authorization bypass."""

from dataclasses import replace
from secrets import token_urlsafe

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.main import create_app


@pytest.fixture
def dev_settings(tmp_path, monkeypatch):
    # Model the supported local launcher even when this unit suite itself runs
    # in a verification container. Secure Docker rejection is tested separately.
    from types import SimpleNamespace
    monkeypatch.setattr("app.main.Path", lambda path: SimpleNamespace(exists=lambda: False))
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'docs.db'}",
        artifact_storage_dir=tmp_path / "artifacts",
        app_environment="development",
        enable_dev_swagger=True,
        secure_container=False,
        admin_api_key=token_urlsafe(32),
        operator_api_key=token_urlsafe(32),
    )


def test_local_browser_can_load_swagger_schema_and_redirect_without_credentials(
    dev_settings,
):
    with TestClient(
        create_app(dev_settings),
        base_url="http://127.0.0.1:8000",
        client=("127.0.0.1", 1234),
    ) as client:
        page = client.get("/docs")
        assert page.status_code == 200 and "SwaggerUIBundle" in page.text
        assert dev_settings.admin_api_key not in page.text
        assert client.get("/docs/oauth2-redirect").status_code == 200
        spec = client.get("/openapi.json").json()
        assert (
            spec["components"]["securitySchemes"]["APIKeyHeader"]["name"] == "X-API-Key"
        )
        assert client.get("/metrics").status_code == 401
        assert (
            client.put(
                "/routing/model", json={"weights": {"v1": 100}, "expected_revision": 0}
            ).status_code
            == 401
        )
        assert (
            client.put(
                "/routing/model",
                headers={"X-API-Key": dev_settings.operator_api_key},
                json={"weights": {"v1": 100}, "expected_revision": 0},
            ).status_code
            == 403
        )
        assert (
            client.get(
                "/metrics", headers={"X-API-Key": dev_settings.operator_api_key}
            ).status_code
            == 200
        )
        assert (
            client.get(
                "/deployments/missing/health",
                headers={"X-API-Key": dev_settings.operator_api_key},
            ).status_code
            == 403
        )


@pytest.mark.parametrize(
    "peer,host,extra",
    [
        ("172.18.0.1", "127.0.0.1", {"X-Forwarded-For": "127.0.0.1"}),
        ("203.0.113.1", "localhost", {}),
        ("127.0.0.1", "evil.example", {}),
        ("127.0.0.1", "127.0.0.1", {"Origin": "https://evil.example"}),
        ("127.0.0.1", "127.0.0.1", {"Origin": "https://[invalid"}),
        ("127.0.0.1", "localhost", {"Sec-Fetch-Site": "cross-site"}),
    ],
)
def test_development_docs_reject_nonlocal_browser_context(
    dev_settings, peer, host, extra
):
    with TestClient(
        create_app(dev_settings),
        base_url="http://" + host,
        client=(peer, 1234),
        headers=extra,
    ) as client:
        for path in ["/docs", "/openapi.json", "/docs/oauth2-redirect"]:
            assert client.get(path).status_code == 403


def test_docs_stay_protected_when_disabled(dev_settings):
    with TestClient(
        create_app(replace(dev_settings, enable_dev_swagger=False)),
        client=("127.0.0.1", 1234),
    ) as client:
        for path in ["/docs", "/openapi.json", "/docs/oauth2-redirect"]:
            assert client.get(path).status_code == 401


@pytest.mark.parametrize(
    "values",
    [
        {"app_environment": "secure"},
        {"secure_container": True},
        {"enable_dev_swagger": "true"},
        {"app_environment": "production"},
    ],
)
def test_unsafe_development_configuration_is_rejected(dev_settings, values):
    with pytest.raises(ValueError):
        replace(dev_settings, **values)


def test_actual_container_cannot_enable_development_swagger(dev_settings, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("app.main.Path", lambda path: SimpleNamespace(exists=lambda: True))
    with pytest.raises(ValueError, match="unsupported inside Docker"):
        create_app(dev_settings)
