"""Binary artifact storage independent of physical filesystem locations."""

from abc import ABC, abstractmethod


class ArtifactStorage(ABC):
    """Logical keys are opaque to consumers; content is never deserialized.

    Saves are create-only: duplicate keys raise DuplicateArtifactError and must
    leave existing content intact. A successful save publishes complete bytes.
    An ordinary failed save publishes nothing; an exceptional cleanup failure
    must report its uncertain state explicitly (ArtifactCleanupError).
    Get/delete raise ArtifactNotFoundError for missing keys. Invalid keys raise
    DomainValidationError; other backend failures raise ArtifactStorageError.
    """

    @abstractmethod
    def save(self, key: str, content: bytes) -> None:
        """Create an artifact without replacing an existing key."""

    @abstractmethod
    def retrieve(self, key: str) -> bytes:
        """Return the exact artifact bytes."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete an artifact, raising if it is missing."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether a valid key has an artifact."""
