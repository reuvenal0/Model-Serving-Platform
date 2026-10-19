"""Loading contract using the bytes returned by ArtifactStorage.retrieve()."""

from abc import ABC, abstractmethod

from app.interfaces.loaded_model import LoadedModel


class ModelLoader(ABC):
    """Deserialize trusted artifact bytes into a reusable runtime.

    Loaders know neither repositories nor storage paths. A loader may be called
    concurrently for different versions and must keep per-load state local.
    Ordinary loading failures raise ModelLoadingError and preserve their cause.
    """

    @abstractmethod
    def load(self, artifact: bytes) -> LoadedModel:
        """Load once; predictions reuse the returned runtime."""
