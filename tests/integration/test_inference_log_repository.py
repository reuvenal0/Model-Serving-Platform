"""Persistent bounded logs and exact SQL aggregate calculations."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.orm import sessionmaker

from app.core import exceptions as errors
from app.domain.inference_log import InferenceLog
from app.infrastructure.database.database import create_sqlite_engine, initialize_database
from app.infrastructure.database.records import InferenceLogRecord
from app.infrastructure.database.sqlite_inference_log_repository import SQLiteInferenceLogRepository
from app.services.metrics_collector import MetricsCollector

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def logs(engine):
    return SQLiteInferenceLogRepository(sessionmaker(engine))


@pytest.fixture
def filters():
    return MetricsCollector.filters(start=START, end=START+timedelta(seconds=100))


def observation(identifier='one', latency=1, success=True, version='v1', model='model', timestamp=START):
    return InferenceLog(identifier, model, version, latency, success,
                        None if success else 'prediction_failed', timestamp=timestamp)


def test_round_trip_unique_id_and_persistence(logs, filters, database_url):
    log = observation(timestamp=START.astimezone(timezone(timedelta(hours=2))))
    logs.append_log(log)
    with pytest.raises(errors.DuplicateInferenceLogError):
        logs.append_log(replace(log, latency_ms=99))
    engine = create_sqlite_engine(database_url)
    try:
        other = SQLiteInferenceLogRepository(sessionmaker(engine))
        page = other.query_logs(filters)
        assert page.logs == (log,) and page.next_cursor is None
        assert page.logs[0].timestamp.tzinfo is timezone.utc
    finally:
        engine.dispose()


def test_filters_and_stable_keyset_pagination(logs, filters):
    for i in range(5):
        logs.append_log(observation(str(i), success=i%2 == 0, version='v1' if i<3 else 'v2'))
    logs.append_log(observation('other', model='other'))
    first = logs.query_logs(filters, limit=2)
    second = logs.query_logs(filters, limit=2, cursor=first.next_cursor)
    third = logs.query_logs(filters, limit=2, cursor=second.next_cursor)
    assert [item.request_id for item in (*first.logs, *second.logs, *third.logs)] == ['0','1','2','3','4','other']
    assert third.next_cursor is None
    assert [log.request_id for log in logs.query_logs(replace(filters, version='v2')).logs] == ['3','4']
    assert len(logs.query_logs(replace(filters, model_name='model', success=False)).logs) == 2
    assert not logs.query_logs(replace(filters, model_name='missing')).logs


@pytest.mark.parametrize('limit,cursor', [(0,0), (201,0), (True,0), (1,-1), (1,True)])
def test_pagination_limits(logs, filters, limit, cursor):
    with pytest.raises(errors.DomainValidationError):
        logs.query_logs(filters, limit=limit, cursor=cursor)


def test_time_boundary_is_start_inclusive_end_exclusive(logs, filters):
    for name, timestamp in [('before', filters.start-timedelta(microseconds=1)),
                            ('start', filters.start), ('inside', filters.end-timedelta(microseconds=1)),
                            ('end', filters.end)]:
        logs.append_log(observation(name, timestamp=timestamp))
    assert [log.request_id for log in logs.query_logs(filters).logs] == ['start', 'inside']
    assert MetricsCollector(logs).get_metrics(filters).total_requests == 2


def test_empty_and_failure_only_metrics(logs, filters):
    collector = MetricsCollector(logs)
    empty = collector.get_metrics(filters)
    assert empty.total_requests == 0 and empty.error_rate == 0 and empty.throughput_requests_per_second == 0
    assert empty.average_latency_ms is None and empty.p50_latency_ms is None
    assert collector.get_version_metrics(filters) == []
    logs.append_log(observation(success=False, version=None))
    failed = collector.get_metrics(filters)
    assert failed.failed_requests == 1 and failed.error_rate == 1
    assert failed.average_latency_ms is None
    assert collector.get_version_metrics(filters)[0].version is None
    assert logs.query_logs(replace(filters, unknown_version=True)).logs[0].version is None


def test_known_percentiles_error_rate_throughput_and_version_groups(logs, filters):
    for value in range(1,101):
        logs.append_log(observation(str(value), latency=value))
    for value in range(5):
        logs.append_log(observation(f'failure-{value}', latency=10000, success=False, version='v2'))
    collector = MetricsCollector(logs)
    metrics = collector.get_metrics(filters)
    assert (metrics.total_requests, metrics.successful_requests, metrics.failed_requests) == (105,100,5)
    assert metrics.error_rate == pytest.approx(5/105)
    assert metrics.throughput_requests_per_second == pytest.approx(1.05)
    assert metrics.average_latency_ms == pytest.approx(50.5)
    assert (metrics.p50_latency_ms, metrics.p95_latency_ms, metrics.p99_latency_ms) == (50,95,99)
    groups = collector.get_version_metrics(filters)
    assert [group.version for group in groups] == ['v1','v2']
    assert groups[0].successful_requests == 100 and groups[1].failed_requests == 5
    assert groups[1].p99_latency_ms is None
    assert collector.get_metrics(replace(filters, version='v1')).failed_requests == 0


@pytest.mark.parametrize('values,expected', [([7], (7,7,7)), ([1,2], (1,2,2)), ([0,0,0], (0,0,0)),
                                            ([1,4,9,16], (4,16,16))])
def test_small_nearest_rank_datasets(logs, filters, values, expected):
    for index, value in enumerate(values):
        logs.append_log(observation(str(index), value))
    metrics = MetricsCollector(logs).get_metrics(filters)
    assert (metrics.p50_latency_ms, metrics.p95_latency_ms, metrics.p99_latency_ms) == expected


def test_numeric_stability_for_large_finite_values(logs, filters):
    for i in range(5):
        logs.append_log(observation(str(i), 1e308))
    assert MetricsCollector(logs).get_metrics(filters).average_latency_ms == pytest.approx(1e308)


def test_concurrent_append_has_unique_records(logs, filters):
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: logs.append_log(observation(str(i))), range(40)))
    assert MetricsCollector(logs).get_metrics(filters).total_requests == 40
    assert len({log.request_id for log in logs.query_logs(filters).logs}) == 40


def test_aggregation_never_constructs_domain_logs_and_uses_bounded_sql(logs, filters, engine):
    with engine.begin() as connection:
        connection.execute(InferenceLogRecord.__table__.insert(), [dict(
            request_id=str(i), model_name='model', version='v1', latency_ms=float(i), success=True,
            error_code=None, timestamp=START, input_hash=None,
        ) for i in range(10000)])
    with patch.object(logs, '_domain', side_effect=AssertionError('materialized a log')):
        metrics = MetricsCollector(logs).get_metrics(filters)
        assert metrics.total_requests == 10000 and metrics.p95_latency_ms == 9499
        assert metrics.average_latency_ms == pytest.approx(4999.5)
    statements = []
    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)
    event.listen(engine, 'before_cursor_execute', capture)
    try:
        assert len(logs.query_logs(filters, limit=3).logs) == 3
    finally:
        event.remove(engine, 'before_cursor_execute', capture)
    assert any('LIMIT' in statement for statement in statements)


def test_raw_error_messages_rejected_and_mutable_entities_revalidated(logs, filters):
    with pytest.raises(errors.DomainValidationError):
        logs.append_log(replace(observation(success=False), error_message='/private/artifacts/password'))
    mutated = observation()
    mutated.latency_ms = -1
    with pytest.raises(errors.DomainValidationError):
        logs.append_log(mutated)
    assert not logs.query_logs(filters).logs


def test_invalid_stored_latency_cannot_bypass_domain_validation(logs, engine, filters):
    logs.append_log(observation())
    with engine.begin() as connection:
        connection.execute(text("UPDATE inference_logs SET latency_ms = :value"), {'value': float('inf')})
    with pytest.raises(errors.RepositoryError):
        logs.query_logs(filters)


def test_group_count_is_bounded_and_requires_narrower_filter(logs, filters, engine):
    with engine.begin() as connection:
        connection.execute(InferenceLogRecord.__table__.insert(), [dict(
            request_id=str(i), model_name='model', version=f'v{i}', latency_ms=1.0,
            success=True, error_code=None, timestamp=START, input_hash=None,
        ) for i in range(201)])
    with pytest.raises(errors.DomainValidationError, match='200'):
        MetricsCollector(logs).get_version_metrics(filters)
    assert len(MetricsCollector(logs).get_version_metrics(replace(filters, version='v0'))) == 1


def test_cursor_larger_than_sqlite_integer_is_rejected(logs, filters):
    with pytest.raises(errors.DomainValidationError):
        logs.query_logs(filters, cursor=2**80)
