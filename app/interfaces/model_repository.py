"""Insert-only metadata persistence contract expressed in domain types."""

from abc import ABC, abstractmethod

from app.domain.model import Model
from app.domain.model_version import ModelVersion


class ModelRepository(ABC):
    """Saves commit before returning and roll back on failure; no upserts.

    Get/delete raise the corresponding Model[Version]NotFoundError. Saves raise
    DuplicateModelError or DuplicateVersionError, and save_version requires a
    parent model. Lists are sorted by name/version; list_versions requires an
    existing model (an existing model with no versions returns []). Other
    persistence failures raise RepositoryError. Returned values are detached
    domain snapshots, never persistence objects.
    """

    @abstractmethod
    def save_model(self, model: Model) -> None:
        """Insert a model, never overwrite an existing model."""

    @abstractmethod
    def get_model(self, name: str) -> Model:
        """Return a model or raise ModelNotFoundError."""

    @abstractmethod
    def list_models(self) -> list[Model]:
        """Return all models ordered by name."""

    @abstractmethod
    def save_version(self, version: ModelVersion) -> None:
        """Insert a version under an existing model."""

    @abstractmethod
    def get_version(self, model_name: str, version: str) -> ModelVersion:
        """Return a version or raise ModelVersionNotFoundError."""

    @abstractmethod
    def list_versions(self, model_name: str) -> list[ModelVersion]:
        """Return versions ordered lexically by version identifier."""

    @abstractmethod
    def delete_version(self, model_name: str, version: str) -> ModelVersion:
        """Atomically delete and return its snapshot, preserving the parent model."""
