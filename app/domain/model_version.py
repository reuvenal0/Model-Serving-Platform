"""Registration metadata for a specific model version."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.domain._validation import enum_value, nonempty, utc_now, utc_timestamp
from app.domain.enums import ModelFramework, ModelVersionStatus


@dataclass
class ModelVersion:
    model_name: str
    version: str
    artifact_location: str
    framework: ModelFramework
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    status: ModelVersionStatus = ModelVersionStatus.REGISTERED
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        nonempty(self.model_name, "model_name")
        nonempty(self.version, "version")
        nonempty(self.artifact_location, "artifact_location")
        enum_value(self.framework, ModelFramework, "framework")
        enum_value(self.status, ModelVersionStatus, "status")
        self.created_at = utc_timestamp(self.created_at, "created_at")
