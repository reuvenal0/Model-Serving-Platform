"""Contract for a reusable model already loaded into application memory."""

from typing import Any, Protocol


class LoadedModel(Protocol):
    """Predict without storage or loading responsibilities.

    Input/output structures are defined by each adapter, not normalized to JSON.
    Runtimes may be shared by many callers and must protect prediction state.
    The provided adapters serialize predictions per runtime. Callers must not
    mutate input buffers during prediction or access private backend objects.
    """

    def predict(self, inputs: Any) -> Any:
        """Return predictions or raise PredictionError with the backend cause."""
        ...
