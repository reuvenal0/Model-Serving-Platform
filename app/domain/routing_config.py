"""Validated traffic distribution, independent of routing algorithms."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType

from app.core.exceptions import DomainValidationError
from app.domain._validation import nonempty, utc_now, utc_timestamp


@dataclass(frozen=True)
class RoutingConfig:
    """Read-only public configuration replaced through update_weights()."""

    __hash__ = None

    model_name: str
    weights: Mapping[str, int]
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        nonempty(self.model_name, "model_name")
        weights = self._validated_weights(self.weights)
        timestamp = utc_timestamp(self.updated_at, "updated_at")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "updated_at", timestamp)

    @staticmethod
    def _validated_weights(weights: Mapping[str, int]) -> Mapping[str, int]:
        if not isinstance(weights, Mapping) or not weights:
            raise DomainValidationError("weights must be a nonempty mapping")
        replacement = dict(weights)
        for version, weight in replacement.items():
            nonempty(version, "version")
            if type(weight) is not int or not 0 <= weight <= 100:
                raise DomainValidationError("weights must be integers between 0 and 100")
        if sum(replacement.values()) != 100:
            raise DomainValidationError("routing weights must sum to 100")
        return MappingProxyType(replacement)

    def update_weights(self, weights: Mapping[str, int]) -> None:
        """Validate and copy a complete replacement before changing state.

        Invalid input preserves both the prior weights and updated_at. This is
        validation atomicity, not synchronization between concurrent writers.
        """
        replacement = self._validated_weights(weights)
        timestamp = utc_now()
        object.__setattr__(self, "weights", replacement)
        object.__setattr__(self, "updated_at", timestamp)
