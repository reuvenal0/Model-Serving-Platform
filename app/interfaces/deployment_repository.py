"""Deployment persistence with explicit optimistic concurrency tokens."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.domain.deployment import Deployment


@dataclass(frozen=True)
class DeploymentSnapshot:
    """Detached domain entity plus a backend-independent revision token.

    The entity can change through its domain methods; retain its read revision
    when updating. Tokens are scoped to the deployment ID and cannot be reused.
    """

    deployment: Deployment
    revision: int


class DeploymentRepository(ABC):
    """Commit each write before returning; never expose ORM records.

    Save is insert-only (revision 0); duplicate IDs raise DuplicateDeploymentError.
    Get/update of missing IDs raise DeploymentNotFoundError. Updates require the
    last observed revision and preserve model identity/created_at. Stale writes
    raise DeploymentConflictError. Other failures raise DeploymentPersistenceError.
    Version references must exist and history must protect them against deletion.
    Transitions belong to Deployment and the manager, not this repository.
    """

    @abstractmethod
    def save_deployment(self, deployment: Deployment) -> DeploymentSnapshot:
        """Insert and return the validated snapshot at revision 0."""

    @abstractmethod
    def get_deployment(self, deployment_id: str) -> DeploymentSnapshot:
        """Retrieve a detached snapshot, or raise DeploymentNotFoundError."""

    @abstractmethod
    def list_deployments(self, model_name: str | None = None) -> list[DeploymentSnapshot]:
        """List by ID, optionally filtered; no matches returns an empty list."""

    @abstractmethod
    def update_deployment(
        self, deployment: Deployment, *, expected_revision: int,
    ) -> DeploymentSnapshot:
        """Compare-and-swap a snapshot, incrementing the revision on success."""
