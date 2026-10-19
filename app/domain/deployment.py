"""Logical deployment state, independent of execution and orchestration."""

from dataclasses import dataclass, field
from datetime import datetime

from app.core.exceptions import DomainValidationError, InvalidDeploymentStateError
from app.domain._validation import enum_value, nonempty, utc_now, utc_timestamp
from app.domain.enums import DeploymentStatus
from app.domain.resource_config import ResourceConfig


@dataclass(frozen=True)
class Deployment:
    """Public fields are read-only; domain operations control state changes.

    Construction may restore any valid status. Operations record logical state
    only; a future manager performs external work before reporting its result.
    Stopped deployments cannot restart; retry/redeployment policy is deferred.
    """

    # Domain operations mutate state, so instances must not be hashable.
    __hash__ = None

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
        self._validate_replicas(self.replicas)
        if not isinstance(self.resources, ResourceConfig):
            raise DomainValidationError("resources must be a ResourceConfig")
        enum_value(self.status, DeploymentStatus, "status")
        object.__setattr__(self, "created_at", utc_timestamp(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", utc_timestamp(self.updated_at, "updated_at"))

    @staticmethod
    def _validate_replicas(replicas: int) -> None:
        if type(replicas) is not int or replicas < 0:
            raise DomainValidationError("replicas must be a nonnegative integer")

    def _transition(
        self, target: DeploymentStatus, allowed: tuple[DeploymentStatus, ...],
    ) -> None:
        if self.status not in allowed:
            raise InvalidDeploymentStateError(
                f"Cannot transition deployment from {self.status.value} to {target.value}"
            )
        timestamp = utc_now()
        object.__setattr__(self, "status", target)
        object.__setattr__(self, "updated_at", timestamp)

    def activate(self) -> None:
        """Record successful activation of a deploying deployment."""
        self._transition(DeploymentStatus.RUNNING, (DeploymentStatus.DEPLOYING,))

    def fail(self) -> None:
        """Record deployment or runtime failure."""
        self._transition(
            DeploymentStatus.FAILED,
            (DeploymentStatus.DEPLOYING, DeploymentStatus.RUNNING),
        )

    def stop(self) -> None:
        """Record completed stopping/cleanup from any non-stopped state."""
        self._transition(
            DeploymentStatus.STOPPED,
            (DeploymentStatus.DEPLOYING, DeploymentStatus.RUNNING, DeploymentStatus.FAILED),
        )

    def scale(self, replicas: int) -> None:
        """Update the logical replica count while deploying or running."""
        self._validate_replicas(replicas)
        if self.status not in (DeploymentStatus.DEPLOYING, DeploymentStatus.RUNNING):
            raise InvalidDeploymentStateError(
                f"Cannot scale a deployment in state {self.status.value}"
            )
        timestamp = utc_now()
        object.__setattr__(self, "replicas", replicas)
        object.__setattr__(self, "updated_at", timestamp)
