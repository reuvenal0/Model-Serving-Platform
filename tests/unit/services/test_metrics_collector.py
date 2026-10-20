"""Window validation, classification, service boundaries, and empty contracts."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from app.core import exceptions as errors
from app.domain.inference_log import InferenceLog
from app.interfaces.inference_log_repository import InferenceLogRepository, LogFilters, MetricAggregate
from app.services.metrics_collector import MetricsCollector, classify_error

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize('model,version', [('model', None), (None, None), (None, 'v1')])
def test_unknown_failed_identifiers_are_explicit(model, version):
    log = InferenceLog('id', model, version, 1, False)
    assert (log.model_name, log.version) == (model, version)


@pytest.mark.parametrize('model,version', [('model', None), (None, None), (None, 'v1')])
def test_success_requires_known_identifiers(model, version):
    with pytest.raises(errors.DomainValidationError):
        InferenceLog('id', model, version, 1, True)


@pytest.mark.parametrize('kwargs', [
    {'start': NOW, 'end': NOW}, {'start': NOW, 'end': NOW-timedelta(seconds=1)},
    {'start': NOW, 'end': NOW+timedelta(days=32)}, {'start': NOW.replace(tzinfo=None), 'end': NOW},
    {'start': NOW, 'end': NOW+timedelta(seconds=1), 'version': ' '},
    {'start': NOW, 'end': NOW+timedelta(seconds=1), 'success': 1},
    {'start': NOW, 'end': NOW+timedelta(seconds=1), 'version': 'v1', 'unknown_version': True},
])
def test_invalid_filters(kwargs):
    with pytest.raises(errors.DomainValidationError):
        LogFilters(**kwargs)


def test_default_window_and_timezone_normalization():
    filters = MetricsCollector.filters(end=NOW)
    assert filters.start == NOW-timedelta(minutes=5)
    shifted = NOW.astimezone(timezone(timedelta(hours=2)))
    assert MetricsCollector.filters(end=shifted).end == NOW


@pytest.mark.parametrize('error,code', [
    (errors.PredictionInputError('secret'), 'invalid_input'),
    (errors.DomainValidationError('secret'), 'invalid_input'),
    (errors.RoutingNotFoundError('secret'), 'routing_not_found'),
    (errors.RoutingUnavailableError('secret'), 'routing_unavailable'),
    (errors.ModelVersionNotFoundError('secret'), 'not_found'),
    (errors.DeploymentConflictError('secret'), 'deployment_conflict'),
    (errors.ModelLoadingError('secret'), 'runtime_loading_failed'),
    (errors.ArtifactNotFoundError('secret'), 'artifact_unavailable'),
    (errors.PredictionSerializationError('secret'), 'serialization_failed'),
    (errors.PredictionError('secret'), 'prediction_failed'),
    (errors.ModelContractError('secret'), 'model_contract_invalid'),
    (errors.RepositoryError('secret'), 'persistence_unavailable'),
    (errors.InferenceUnavailableError('secret'), 'unavailable'),
    (errors.UnsupportedFrameworkError('secret'), 'unsupported_framework'),
    (RuntimeError('secret'), 'internal_error'),
])
def test_sanitized_classification(error, code):
    assert classify_error(error) == code


def test_collector_uses_storage_aggregates_and_preserves_write_failures():
    repository = Mock(spec=InferenceLogRepository)
    repository.aggregate_metrics.return_value = [MetricAggregate(None, 4, 3, 2, 2, 3, 3)]
    collector = MetricsCollector(repository)
    filters = collector.filters(end=NOW)
    metrics = collector.get_metrics(filters)
    assert metrics.total_requests == 4 and metrics.failed_requests == 1
    assert metrics.error_rate == 0.25 and metrics.throughput_requests_per_second == 4/300
    repository.query_logs.assert_not_called()
    failure = errors.RepositoryError('private backend')
    repository.append_log.side_effect = failure
    with pytest.raises(errors.RepositoryError) as caught:
        collector.record(InferenceLog('id', 'model', 'v1', 1, True))
    assert caught.value is failure


def test_default_window_rejects_datetime_underflow():
    with pytest.raises(errors.DomainValidationError) as caught:
        MetricsCollector.filters(end=datetime.min.replace(tzinfo=timezone.utc))
    assert isinstance(caught.value.__cause__, OverflowError)


@pytest.mark.parametrize('start,end', [
    (datetime(1,1,1,tzinfo=timezone(timedelta(hours=1))),
     datetime(1,1,1,0,5,tzinfo=timezone(timedelta(hours=1)))),
    (datetime(9999,12,31,23,54,tzinfo=timezone(timedelta(hours=-1))),
     datetime(9999,12,31,23,59,tzinfo=timezone(timedelta(hours=-1)))),
])
def test_utc_normalization_overflow_is_a_validation_error(start, end):
    with pytest.raises(errors.DomainValidationError) as caught:
        MetricsCollector.filters(start=start, end=end)
    assert isinstance(caught.value.__cause__, OverflowError)
