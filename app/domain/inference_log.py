"""Inference request metadata without raw input payloads."""

from dataclasses import dataclass, field
from datetime import datetime

from app.core.exceptions import DomainValidationError
from app.domain._validation import finite_number, nonempty, utc_now, utc_timestamp


@dataclass
class InferenceLog:
    request_id: str
    model_name: str
    version: str
    latency_ms: float
    success: bool
    error_message: str | None = None
    timestamp: datetime = field(default_factory=utc_now)
    input_hash: str | None = None

    def __post_init__(self) -> None:
        nonempty(self.request_id, "request_id")
        nonempty(self.model_name, "model_name")
        nonempty(self.version, "version")
        finite_number(self.latency_ms, "latency_ms")
        if self.latency_ms < 0:
            raise DomainValidationError("latency_ms cannot be negative")
        if not isinstance(self.success, bool):
            raise DomainValidationError("success must be a boolean")
        self.timestamp = utc_timestamp(self.timestamp, "timestamp")
