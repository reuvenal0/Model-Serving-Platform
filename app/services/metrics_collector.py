"""Finalized observation writes and precise windowed metrics; no runtime state."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.core import exceptions as errors
from app.domain._validation import utc_now
from app.domain.inference_log import InferenceLog
from app.interfaces.inference_log_repository import InferenceLogRepository, LogFilters, LogPage, MetricAggregate
from app.core.exceptions import DomainValidationError


@dataclass(frozen=True)
class InferenceMetrics:
    model_name: str | None
    version: str | None
    start: datetime
    end: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float
    throughput_requests_per_second: float
    average_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None


def classify_error(error: Exception) -> str:
    # Specific subclasses precede their broader application error categories.
    classifications = (
        ((errors.PredictionInputError, errors.DomainValidationError), 'invalid_input'),
        ((errors.RoutingNotFoundError, errors.RoutingHistoryNotFoundError), 'routing_not_found'),
        ((errors.RoutingUnavailableError,), 'routing_unavailable'),
        ((errors.ModelNotFoundError, errors.ModelVersionNotFoundError, errors.DeploymentNotFoundError), 'not_found'),
        ((errors.DeploymentConflictError, errors.InvalidDeploymentStateError, errors.RoutingConflictError), 'deployment_conflict'),
        ((errors.PredictionSerializationError,), 'serialization_failed'),
        ((errors.ModelLoadingError,), 'runtime_loading_failed'),
        ((errors.ArtifactStorageError,), 'artifact_unavailable'),
        ((errors.ModelContractError,), 'model_contract_invalid'),
        ((errors.RepositoryError,), 'persistence_unavailable'),
        ((errors.UnsupportedFrameworkError,), 'unsupported_framework'),
        ((errors.InferenceUnavailableError, errors.DeploymentHealthCheckError), 'unavailable'),
        ((errors.PredictionError,), 'prediction_failed'),
    )
    return next((code for types, code in classifications if isinstance(error, types)), 'internal_error')


class MetricsCollector:
    """Persistence errors propagate to callers; inference applies best effort.

    Read endpoints propagate errors normally rather than claiming empty metrics
    on backend failure. No unbounded in-memory observations or counters are kept.
    """

    def __init__(self, repository: InferenceLogRepository) -> None:
        self._repository = repository

    @staticmethod
    def filters(*, start: datetime | None = None, end: datetime | None = None,
                model_name: str | None = None, version: str | None = None,
                success: bool | None = None, unknown_version: bool = False) -> LogFilters:
        end = end if end is not None else utc_now()

        if start is None:
            try:
                start = end - timedelta(minutes=5)
            except OverflowError as exc:
                raise DomainValidationError("The requested time window is outside the supported datetime range") from exc

        return LogFilters(start, end, model_name, version, success, unknown_version)

    def record(self, observation: InferenceLog) -> None:
        self._repository.append_log(observation)

    @staticmethod
    def _metrics(filters: LogFilters, row: MetricAggregate) -> InferenceMetrics:
        failed = row.total_requests - row.successful_requests
        return InferenceMetrics(filters.model_name, row.version, filters.start, filters.end,
                                row.total_requests, row.successful_requests, failed,
                                failed / row.total_requests if row.total_requests else 0.0,
                                row.total_requests / (filters.end - filters.start).total_seconds(),
                                row.average_latency_ms, row.p50_latency_ms, row.p95_latency_ms, row.p99_latency_ms)

    def get_metrics(self, filters: LogFilters | None = None) -> InferenceMetrics:
        filters = filters if filters is not None else self.filters()
        return self._metrics(filters, self._repository.aggregate_metrics(filters)[0])

    def get_version_metrics(self, filters: LogFilters) -> list[InferenceMetrics]:
        return [self._metrics(filters, row) for row in self._repository.aggregate_metrics(filters, by_version=True)]

    def query_logs(self, filters: LogFilters | None = None, *, limit: int = 50, cursor: int = 0) -> LogPage:
        return self._repository.query_logs(filters if filters is not None else self.filters(), limit=limit, cursor=cursor)