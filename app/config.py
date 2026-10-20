"""Minimal environment-backed settings; importing this module has no side effects."""

from dataclasses import dataclass, field
import os
import math
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
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "sqlite:///./model_serving.db"
        )
    )
    max_upload_size_bytes: int = field(
        default_factory=lambda: int(
            os.getenv("MAX_UPLOAD_SIZE_BYTES", str(100 * 1024 * 1024))
        )
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
    max_runtime_artifact_bytes: int = field(
        default_factory=lambda: int(
            os.getenv("MAX_RUNTIME_ARTIFACT_BYTES", str(100 * 1024 * 1024))
        )
    )
    operator_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPERATOR_API_KEY"), repr=False
    )
    admin_api_key: str | None = field(
        default_factory=lambda: os.getenv("ADMIN_API_KEY"), repr=False
    )
    max_concurrent_inference: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONCURRENT_INFERENCE", "4"))
    )
    max_waiting_inference: int = field(
        default_factory=lambda: int(os.getenv("MAX_WAITING_INFERENCE", "8"))
    )
    admission_wait_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("ADMISSION_WAIT_TIMEOUT_SECONDS", "5"))
    )
    enable_dev_swagger: bool = field(
        default_factory=lambda: _env_bool("ENABLE_DEV_SWAGGER")
    )
    app_environment: str = field(default_factory=lambda: os.getenv("APP_ENV", "secure"))
    secure_container: bool = field(
        default_factory=lambda: _env_bool("MSP_SECURE_CONTAINER")
    )

    def __post_init__(self) -> None:
        if (
            type(self.enable_dev_swagger) is not bool
            or type(self.secure_container) is not bool
        ):
            raise ValueError("Swagger/container flags must be boolean")
        if self.app_environment not in ("secure", "development"):
            raise ValueError("APP_ENV must be secure or development")
        if self.enable_dev_swagger and (
            self.app_environment != "development" or self.secure_container
        ):
            raise ValueError(
                "Development Swagger requires APP_ENV=development outside secure Docker"
            )
        for name in ("operator_api_key", "admin_api_key"):
            key = getattr(self, name)
            if key is not None and (
                not isinstance(key, str)
                or not 32 <= len(key) <= 256
                or any(not 33 <= ord(char) <= 126 for char in key)
            ):
                raise ValueError(
                    f"{name} must contain 32 to 256 visible ASCII characters without spaces"
                )
        if (
            self.admin_api_key is not None
            and self.admin_api_key == self.operator_api_key
        ):
            raise ValueError("Administrative and operator keys must differ")
        if (
            type(self.max_concurrent_inference) is not int
            or self.max_concurrent_inference <= 0
        ):
            raise ValueError("max_concurrent_inference must be a positive integer")
        if (
            type(self.max_waiting_inference) is not int
            or self.max_waiting_inference < 0
        ):
            raise ValueError("max_waiting_inference must be a nonnegative integer")
        timeout = self.admission_wait_timeout_seconds
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                "admission_wait_timeout_seconds must be finite and positive"
            )
        if (
            type(self.max_upload_size_bytes) is not int
            or self.max_upload_size_bytes <= 0
        ):
            raise ValueError(
                "max_upload_size_bytes must be a strictly positive integer"
            )
        if (
            type(self.runtime_cache_max_size) is not int
            or self.runtime_cache_max_size <= 0
        ):
            raise ValueError(
                "runtime_cache_max_size must be a strictly positive integer"
            )
        if (
            type(self.max_runtime_artifact_bytes) is not int
            or self.max_runtime_artifact_bytes <= 0
        ):
            raise ValueError(
                "max_runtime_artifact_bytes must be a strictly positive integer"
            )
        for name in (
            "max_request_body_bytes",
            "max_batch_size",
            "max_tensor_dimensions",
            "max_tensor_elements",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a strictly positive integer")
        if type(self.trust_sklearn_artifacts) is not bool:
            raise ValueError("trust_sklearn_artifacts must be a boolean")
