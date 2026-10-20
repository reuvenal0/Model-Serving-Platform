"""Bounded keyset log reads and SQL-side exact nearest-rank aggregates."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
import sqlite3

from sqlalchemy import Integer, case, cast, func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.exceptions import DomainValidationError, DuplicateInferenceLogError, RepositoryError
from app.domain.inference_log import InferenceLog
from app.infrastructure.database.records import InferenceLogRecord as Record
from app.interfaces.inference_log_repository import ERROR_CODES, MAX_LOG_CURSOR, InferenceLogRepository, LogFilters, LogPage, MetricAggregate


class SQLiteInferenceLogRepository(InferenceLogRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @contextmanager
    def _transaction(self) -> Iterator[Session]:
        try:
            with self._session_factory.begin() as session:
                yield session
        except IntegrityError as error:
            if getattr(error.orig, 'sqlite_errorcode', None) == sqlite3.SQLITE_CONSTRAINT_UNIQUE:
                raise DuplicateInferenceLogError('Observation ID is already recorded') from error
            raise RepositoryError('Inference observation constraint failed') from error
        except (SQLAlchemyError, ValueError, TypeError) as error:
            raise RepositoryError('Inference observation operation failed') from error

    @staticmethod
    def _conditions(filters: LogFilters):
        conditions = [Record.timestamp >= filters.start, Record.timestamp < filters.end]
        for field in ('model_name', 'version', 'success'):
            value = getattr(filters, field)
            if value is not None:
                conditions.append(getattr(Record, field) == value)
        if filters.unknown_version:
            conditions.append(Record.version.is_(None))
        return conditions

    @staticmethod
    def _domain(record: Record) -> InferenceLog:
        if record.error_code is not None and record.error_code not in ERROR_CODES:
            raise ValueError('Persisted error code is invalid')
        return InferenceLog(record.request_id, record.model_name, record.version, record.latency_ms,
                            record.success, record.error_code, record.timestamp, record.input_hash)

    def append_log(self, log: InferenceLog) -> None:
        log = replace(log)
        if log.error_message is not None and (
                not isinstance(log.error_message, str) or log.error_message not in ERROR_CODES):
            raise DomainValidationError('Persisted errors must be a supported sanitized code')
        if log.success and log.error_message is not None:
            raise DomainValidationError('Successful observations cannot contain an error code')
        with self._transaction() as session:
            session.add(Record(request_id=log.request_id, model_name=log.model_name, version=log.version,
                               latency_ms=log.latency_ms, success=log.success, error_code=log.error_message,
                               timestamp=log.timestamp, input_hash=log.input_hash))

    def query_logs(self, filters: LogFilters, *, limit: int = 50, cursor: int = 0) -> LogPage:
        if (type(limit) is not int or not 1 <= limit <= 200
                or type(cursor) is not int or not 0 <= cursor <= MAX_LOG_CURSOR):
            raise DomainValidationError('Limit must be 1..200 and cursor a nonnegative integer')
        with self._transaction() as session:
            records = list(session.scalars(select(Record).where(*self._conditions(filters), Record.id > cursor)
                                          .order_by(Record.id).limit(limit + 1)))
            next_cursor = records[limit - 1].id if len(records) > limit else None
            return LogPage(tuple(self._domain(record) for record in records[:limit]), next_cursor,
                           filters.start, filters.end)

    def aggregate_metrics(self, filters: LogFilters, *, by_version: bool = False) -> list[MetricAggregate]:
        conditions = self._conditions(filters)
        columns = [Record.version] if by_version else []
        totals = select(*columns, func.count().label('total'),
                        func.sum(case((Record.success.is_(True), 1), else_=0)).label('successful')).where(*conditions)
        if by_version:
            totals = totals.group_by(Record.version).order_by(Record.version).limit(201)
        partition = Record.version if by_version else None
        ranked = select(
            Record.version, Record.latency_ms,
            func.row_number().over(partition_by=partition, order_by=(Record.latency_ms, Record.id)).label('rank'),
            func.count().over(partition_by=partition).label('observations'),
            func.max(Record.latency_ms).over(partition_by=partition).label('maximum'),
        ).where(*conditions, Record.success.is_(True)).subquery()
        # Scaling before AVG avoids SQLite summation overflow for finite but very
        # large observations. No Python log objects are loaded for aggregation.
        divisor = case((ranked.c.maximum > 0, ranked.c.maximum), else_=1.0)
        mean = (func.avg(ranked.c.latency_ms / divisor) * func.max(ranked.c.maximum)).label('average')
        percentile_columns = [func.max(case((
            ranked.c.rank == cast((ranked.c.observations * percent + 99) / 100, Integer), ranked.c.latency_ms,
        ), else_=None)).label(f'p{percent}') for percent in (50, 95, 99)]
        latency_columns = [ranked.c.version] if by_version else []
        latencies = select(*latency_columns, mean, *percentile_columns)
        if by_version:
            latencies = latencies.group_by(ranked.c.version).order_by(ranked.c.version).limit(201)
        with self._transaction() as session:
            counts = list(session.execute(totals).mappings())
            values = {}
            if len(counts) <= 200:
                values = {row['version'] if by_version else None: row
                          for row in session.execute(latencies).mappings()}
        if len(counts) > 200:
            raise DomainValidationError('Narrow the filters to at most 200 version groups')
        result = []
        for count in counts:
            version = count['version'] if by_version else filters.version
            latency = values.get(count['version'] if by_version else None)
            result.append(MetricAggregate(version, count['total'], count['successful'] or 0,
                                          latency['average'] if latency else None,
                                          latency['p50'] if latency else None,
                                          latency['p95'] if latency else None,
                                          latency['p99'] if latency else None))
        return result
