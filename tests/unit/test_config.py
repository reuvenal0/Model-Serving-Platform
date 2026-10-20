from pathlib import Path

import pytest

from app.config import Settings


ENV_KEYS = ("APP_NAME", "ARTIFACT_STORAGE_DIR", "DATABASE_URL", "MAX_UPLOAD_SIZE_BYTES")


def test_default_settings(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    settings = Settings()
    assert settings.app_name == "Model Serving Platform"
    assert settings.artifact_storage_dir == Path("artifacts")
    assert settings.database_url == "sqlite:///./model_serving.db"
    assert settings.max_upload_size_bytes == 100 * 1024 * 1024


def test_environment_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_NAME", "Test Platform")
    monkeypatch.setenv("ARTIFACT_STORAGE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./test.db")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_BYTES", "1024")
    settings = Settings()
    assert settings.app_name == "Test Platform"
    assert settings.artifact_storage_dir == tmp_path / "models"
    assert settings.database_url == "sqlite:///./test.db"
    assert settings.max_upload_size_bytes == 1024
    assert not settings.artifact_storage_dir.exists()


@pytest.mark.parametrize("size", ["0", "-1", "invalid"])
def test_invalid_upload_size(monkeypatch, size):
    monkeypatch.setenv("MAX_UPLOAD_SIZE_BYTES", size)
    with pytest.raises(ValueError):
        Settings()


def test_runtime_cache_default(monkeypatch):
    monkeypatch.delenv("RUNTIME_CACHE_MAX_SIZE", raising=False)
    assert Settings().runtime_cache_max_size == 16


def test_runtime_cache_environment_override(monkeypatch):
    monkeypatch.setenv("RUNTIME_CACHE_MAX_SIZE", "3")
    assert Settings().runtime_cache_max_size == 3
    assert Settings(runtime_cache_max_size=5).runtime_cache_max_size == 5


@pytest.mark.parametrize("value", ["0", "-1", "invalid", "1.5"])
def test_invalid_runtime_cache_environment(monkeypatch, value):
    monkeypatch.setenv("RUNTIME_CACHE_MAX_SIZE", value)
    with pytest.raises(ValueError):
        Settings()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_runtime_cache_constructor(value):
    with pytest.raises(ValueError):
        Settings(runtime_cache_max_size=value)


@pytest.mark.parametrize("name,environment,default", [
    ("max_request_body_bytes", "MAX_REQUEST_BODY_BYTES", 1048576),
    ("max_batch_size", "MAX_BATCH_SIZE", 128),
    ("max_tensor_dimensions", "MAX_TENSOR_DIMENSIONS", 4),
    ("max_tensor_elements", "MAX_TENSOR_ELEMENTS", 100000),
])
def test_inference_limit_defaults_and_environment(monkeypatch, name, environment, default):
    monkeypatch.delenv(environment, raising=False)
    assert getattr(Settings(), name) == default
    monkeypatch.setenv(environment, "2")
    assert getattr(Settings(), name) == 2


@pytest.mark.parametrize("name", ["max_request_body_bytes", "max_batch_size", "max_tensor_dimensions", "max_tensor_elements"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_inference_limits_reject_nonpositive_or_noninteger_values(name, value):
    with pytest.raises(ValueError):
        Settings(**{name: value})


def test_sklearn_api_trust_is_explicit_and_environment_friendly(monkeypatch):
    monkeypatch.delenv("TRUST_SKLEARN_ARTIFACTS", raising=False)
    assert Settings().trust_sklearn_artifacts is False
    monkeypatch.setenv("TRUST_SKLEARN_ARTIFACTS", "true")
    assert Settings().trust_sklearn_artifacts is True
    assert Settings(trust_sklearn_artifacts=False).trust_sklearn_artifacts is False
    monkeypatch.setenv("TRUST_SKLEARN_ARTIFACTS", "yes")
    with pytest.raises(ValueError):
        Settings()
    with pytest.raises(ValueError):
        Settings(trust_sklearn_artifacts=1)
