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
