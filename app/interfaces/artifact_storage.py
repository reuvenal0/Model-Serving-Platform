"""Binary artifact storage independent of physical filesystem locations."""

from abc import ABC, abstractmethod

from app.core.exceptions import ArtifactStorageError, ArtifactTooLargeError, DomainValidationError


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

    @staticmethod
    def _validate_limit(max_bytes: int) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise DomainValidationError("max_bytes must be a positive integer")

    def retrieve_limited(self, key: str, *, max_bytes: int) -> bytes:
        """Return bytes or raise ArtifactTooLargeError before deserialization.

        Compatibility fallback for existing adapters: retrieves then checks.
        Adapters should override this with pre-read size checks and bounded reads;
        LocalArtifactStorage does so. This fallback cannot bound retrieval memory.
        """
        self._validate_limit(max_bytes)
        content = self.retrieve(key)
        if not isinstance(content, bytes):
            raise ArtifactStorageError("Artifact backend did not return bytes")
        if len(content) > max_bytes:
            raise ArtifactTooLargeError("Artifact exceeds maximum runtime loading size")
        return content

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete an artifact, raising if it is missing."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether a valid key has an artifact."""
