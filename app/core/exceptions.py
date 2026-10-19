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
