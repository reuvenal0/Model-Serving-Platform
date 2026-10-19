"""Application-facing errors without transport or persistence dependencies."""


class DomainError(Exception):
    """Base class for errors expressed in domain terms."""


class DomainValidationError(DomainError, ValueError):
    """An entity or value object violates a domain invariant."""


class ModelNotFoundError(DomainError):
    """A requested model does not exist."""


class ModelVersionNotFoundError(DomainError):
    """A requested model version does not exist."""


class DuplicateModelError(DomainError):
    """A model with the same name already exists."""


class DuplicateVersionError(DomainError):
    """A version already exists for the requested model."""


class InvalidDeploymentStateError(DomainError):
    """An operation is incompatible with the deployment state."""


class RepositoryError(DomainError):
    """A metadata backend operation failed; inspect its chained cause."""


class ArtifactStorageError(DomainError):
    """An artifact backend operation failed."""


class ArtifactNotFoundError(ArtifactStorageError):
    """The requested logical artifact key does not exist."""


class DuplicateArtifactError(ArtifactStorageError):
    """The artifact key already exists; its content was preserved."""


class ArtifactCleanupError(ArtifactStorageError):
    """Local temporary-file cleanup failed; manual inspection is required."""

    def __init__(self, key: str, temporary_file: str, published: bool) -> None:
        self.artifact_key = key
        self.temporary_file = temporary_file
        self.published = published
        super().__init__(
            f"Artifact {key!r} temporary cleanup failed at {temporary_file!r}; "
            f"published={published}. Inspect storage before retrying."
        )


class RegistryConsistencyError(DomainError):
    """A cross-backend operation partially failed and requires reconciliation."""

    def __init__(
        self, message: str, artifact_key: str,
        operation_error: Exception | None = None,
        cleanup_error: Exception | None = None,
    ) -> None:
        self.artifact_key = artifact_key
        self.operation_error = operation_error
        self.cleanup_error = cleanup_error
        super().__init__(f"{message} Artifact key: {artifact_key!r}")


class UnsupportedFrameworkError(DomainError):
    """No model loader is configured for the requested framework."""


class ModelLoadingError(DomainError):
    """An artifact could not become a usable loaded model."""


class UntrustedArtifactError(ModelLoadingError):
    """Pickle loading was attempted without explicit source trust."""


class PredictionError(DomainError):
    """Prediction inputs are unsupported or backend execution failed."""
