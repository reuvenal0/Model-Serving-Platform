"""Traffic distribution metadata, independent of routing algorithms."""

from dataclasses import dataclass, field
from datetime import datetime

from app.core.exceptions import DomainValidationError
from app.domain._validation import nonempty, utc_now, utc_timestamp


@dataclass
class RoutingConfig:
    model_name: str
    weights: dict[str, int]
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        nonempty(self.model_name, "model_name")
        if not isinstance(self.weights, dict) or not self.weights:
            raise DomainValidationError("weights must be a nonempty dictionary")
        for version, weight in self.weights.items():
            nonempty(version, "version")
            if type(weight) is not int or not 0 <= weight <= 100:
                raise DomainValidationError("weights must be integers between 0 and 100")
        if sum(self.weights.values()) != 100:
            raise DomainValidationError("routing weights must sum to 100")
        self.weights = self.weights.copy()
        self.updated_at = utc_timestamp(self.updated_at, "updated_at")
