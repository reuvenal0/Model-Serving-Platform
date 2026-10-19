"""Minimal environment-backed settings; importing this module has no side effects."""

from dataclasses import dataclass, field
import os
from pathlib import Path


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

    def __post_init__(self) -> None:
        if type(self.max_upload_size_bytes) is not int or self.max_upload_size_bytes <= 0:
            raise ValueError("max_upload_size_bytes must be a strictly positive integer")
        if type(self.runtime_cache_max_size) is not int or self.runtime_cache_max_size <= 0:
            raise ValueError("runtime_cache_max_size must be a strictly positive integer")
