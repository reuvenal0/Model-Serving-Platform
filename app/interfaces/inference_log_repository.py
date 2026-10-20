"""Bounded observation reads and database-side aggregate contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from app.core.exceptions import DomainValidationError
from app.domain._validation import nonempty, utc_timestamp
from app.domain.inference_log import InferenceLog


ERROR_CODES = frozenset({
    "invalid_input", "not_found", "routing_not_found", "routing_unavailable",
    "deployment_conflict", "unavailable", "runtime_loading_failed", "artifact_unavailable",
    "prediction_failed", "serialization_failed", "model_contract_invalid",
    "persistence_unavailable", "unsupported_framework", "internal_error",
})
MAX_LOG_CURSOR = 2**63 - 1


@dataclass(frozen=True)
class LogFilters:
    """Completion-time window [start, end), at most 31 days, normalized to UTC."""

    start: datetime
    end: datetime
    model_name: str | None = None
    version: str | None = None
    success: bool | None = None
    unknown_version: bool = False

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "start", utc_timestamp(self.start, "start"))
            object.__setattr__(self, "end", utc_timestamp(self.end, "end"))
        except OverflowError as error:
            raise DomainValidationError("Observation timestamps exceed supported UTC datetime bounds") from error
        duration = (self.end - self.start).total_seconds()
        if not 0 < duration <= 31 * 86400:
            raise DomainValidationError("Observation window must be positive and at most 31 days")
        for field in ("model_name", "version"):
            if getattr(self, field) is not None:
                nonempty(getattr(self, field), field)
        if self.success is not None and type(self.success) is not bool:
            raise DomainValidationError("success must be a boolean")
        if type(self.unknown_version) is not bool or (self.unknown_version and self.version is not None):
            raise DomainValidationError("unknown_version cannot be combined with version")


@dataclass(frozen=True)
class LogPage:
    logs: tuple[InferenceLog, ...]
    next_cursor: int | None
    start: datetime
    end: datetime


@dataclass(frozen=True)
class MetricAggregate:
    version: str | None
    total_requests: int
    successful_requests: int
    average_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None


class InferenceLogRepository(ABC):
    """Append commits before returning; duplicate IDs raise DuplicateInferenceLogError.

    Other backend failures raise RepositoryError. No foreign keys to registry
    data: failures may name nonexistent models/versions, and logs are historical
    metadata, not artifact-retention references. Error values are bounded codes.
    """

    @abstractmethod
    def append_log(self, log: InferenceLog) -> None: ...

    @abstractmethod
    def query_logs(self, filters: LogFilters, *, limit: int = 50, cursor: int = 0) -> LogPage:
        """Insertion-ID ascending keyset pagination; limit 1..200, cursor >= 0.

        Empty pages have () and no next cursor. Reuse the exact filters/window
        across pages; this is an append-only feed, not a frozen snapshot.
        """

    @abstractmethod
    def aggregate_metrics(self, filters: LogFilters, *, by_version: bool = False) -> list[MetricAggregate]:
        """Aggregate in storage, not by materializing logs. Latency uses successes.

        Ungrouped empty result contains one zero-count aggregate with null latency;
        grouped empty result is []. Grouped results are version-ordered (null
        first), at most 200 groups; excess groups require a narrower filter.
        All aggregate queries use one consistent read transaction.
        """
