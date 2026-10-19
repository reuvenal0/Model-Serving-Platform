"""Logical deployment metadata without execution or orchestration behavior."""

from dataclasses import dataclass, field
from datetime import datetime

from app.core.exceptions import DomainValidationError
from app.domain._validation import enum_value, nonempty, utc_now, utc_timestamp
from app.domain.enums import DeploymentStatus
from app.domain.resource_config import ResourceConfig


@dataclass
class Deployment:
    id: str
    model_name: str
    version: str
    replicas: int
    resources: ResourceConfig
    status: DeploymentStatus = DeploymentStatus.DEPLOYING
    endpoint: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        nonempty(self.id, "id")
        nonempty(self.model_name, "model_name")
        nonempty(self.version, "version")
        if type(self.replicas) is not int or self.replicas < 0:
            raise DomainValidationError("replicas must be a nonnegative integer")
        if not isinstance(self.resources, ResourceConfig):
            raise DomainValidationError("resources must be a ResourceConfig")
        enum_value(self.status, DeploymentStatus, "status")
        self.created_at = utc_timestamp(self.created_at, "created_at")
        self.updated_at = utc_timestamp(self.updated_at, "updated_at")
