"""Minimal environment-backed settings; importing this module has no side effects."""

from dataclasses import dataclass, field
import os
from pathlib import Path


def _env_bool(name: str, default: str = "false") -> bool:
    value = os.getenv(name, default).lower()
    if value not in ("true", "false"):
        raise ValueError(f"{name} must be true or false")
    return value == "true"


@dataclass(frozen=True)
class Settings:
    app_name: str = field(
        default_factory=lambda: os.getenv("APP_NAME", "Model Serving Platform")
    )
    artifact_storage_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ARTIFACT_STORAGE_DIR", "artifacts"))
    )
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./model_serving.db")
    )
    max_upload_size_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(100 * 1024 * 1024)))
    )

    runtime_cache_max_size: int = field(
        default_factory=lambda: int(os.getenv("RUNTIME_CACHE_MAX_SIZE", "16"))
    )

    max_request_body_bytes: int = field(
        default_factory=lambda: int(os.getenv("MAX_REQUEST_BODY_BYTES", "1048576"))
    )
    max_batch_size: int = field(
        default_factory=lambda: int(os.getenv("MAX_BATCH_SIZE", "128"))
    )
    max_tensor_dimensions: int = field(
        default_factory=lambda: int(os.getenv("MAX_TENSOR_DIMENSIONS", "4"))
    )
    max_tensor_elements: int = field(
        default_factory=lambda: int(os.getenv("MAX_TENSOR_ELEMENTS", "100000"))
    )
    trust_sklearn_artifacts: bool = field(
        default_factory=lambda: _env_bool("TRUST_SKLEARN_ARTIFACTS")
    )

    def __post_init__(self) -> None:
        if type(self.max_upload_size_bytes) is not int or self.max_upload_size_bytes <= 0:
            raise ValueError("max_upload_size_bytes must be a strictly positive integer")
        if type(self.runtime_cache_max_size) is not int or self.runtime_cache_max_size <= 0:
            raise ValueError("runtime_cache_max_size must be a strictly positive integer")
        for name in ("max_request_body_bytes", "max_batch_size", "max_tensor_dimensions", "max_tensor_elements"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a strictly positive integer")
        if type(self.trust_sklearn_artifacts) is not bool:
            raise ValueError("trust_sklearn_artifacts must be a boolean")
