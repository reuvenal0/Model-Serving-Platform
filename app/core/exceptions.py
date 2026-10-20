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


class DeploymentNotFoundError(DomainError):
    """The requested deployment does not exist."""


class DuplicateDeploymentError(DomainError):
    """A deployment ID is already in use."""


class DeploymentPersistenceError(RepositoryError):
    """Deployment metadata could not be read or committed."""


class DeploymentConflictError(DomainError):
    """The snapshot is stale or SQLite requires retrying after a fresh read."""


class DeploymentHealthCheckError(DomainError):
    """Load-readiness or an explicitly configured prediction check failed."""


class DeploymentRecoveryError(DomainError):
    """Lifecycle work failed and recording FAILED also failed."""

    def __init__(
        self, deployment_id: str, operation_error: Exception, persistence_error: Exception,
    ) -> None:
        self.deployment_id = deployment_id
        self.operation_error = operation_error
        self.persistence_error = persistence_error
        super().__init__(
            f"Deployment {deployment_id!r} failed and FAILED could not be recorded; "
            "inspect persisted state and reconcile before treating it as ready"
        )


class ModelVersionInUseError(DomainError):
    """Deployment history references this version; deletion is prohibited."""


class PredictionInputError(PredictionError):
    """Input structure, values, tensor shape, or dtype are unsupported."""


class PredictionSerializationError(PredictionError):
    """Predictions cannot be represented by the supported JSON output contract."""


class InferenceUnavailableError(DomainError):
    """No eligible deployment or application-owned runtime is available."""


class ModelContractError(DomainError):
    """Opt-in inference schema metadata is invalid or unsupported."""
