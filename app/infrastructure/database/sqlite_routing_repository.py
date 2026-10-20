"""Atomic current/history routing persistence with compare-and-swap writes."""

from collections.abc import Iterator
from contextlib import contextmanager
import sqlite3

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.exceptions import (
    DomainValidationError, ModelNotFoundError, ModelVersionNotFoundError, RepositoryError,
    RoutingConflictError, RoutingHistoryNotFoundError, RoutingNotFoundError,
)
from app.domain.routing_config import RoutingConfig
from app.infrastructure.database.records import (
    ModelRecord, ModelVersionRecord, RoutingCurrentRecord, RoutingHistoryRecord, RoutingVersionReferenceRecord,
)
from app.interfaces.routing_repository import RoutingRepository, RoutingSnapshot


def positive_revision(value: int) -> None:
    if type(value) is not int or value < 1:
        raise DomainValidationError("Revision must be a positive integer")


class SQLiteRoutingRepository(RoutingRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @contextmanager
    def _transaction(self) -> Iterator[Session]:
        try:
            with self._session_factory.begin() as session:
                yield session
        except OperationalError as error:
            code = getattr(error.orig, "sqlite_errorcode", 0) or 0
            if code & 0xff in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise RoutingConflictError("SQLite is busy; re-read routing before retrying") from error
            raise RepositoryError("Routing persistence failed") from error
        except IntegrityError as error:
            code = getattr(error.orig, "sqlite_errorcode", 0)
            if code in (sqlite3.SQLITE_CONSTRAINT_UNIQUE, sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY):
                raise RoutingConflictError("Routing revision already exists") from error
            raise RepositoryError("Routing constraint violation") from error
        except (SQLAlchemyError, ValueError, TypeError) as error:
            raise RepositoryError("Routing persistence failed") from error

    @staticmethod
    def _snapshot(record) -> RoutingSnapshot:
        # Domain validation also runs on every read, including corrupt JSON data.
        return RoutingSnapshot(RoutingConfig(record.model_name, record.weights, record.updated_at),
                               record.revision, record.restored_from)

    @staticmethod
    def _current(session: Session, model_name: str) -> RoutingCurrentRecord:
        record = session.get(RoutingCurrentRecord, model_name)
        if record is None:
            raise RoutingNotFoundError("No routing configuration exists")
        return record

    @staticmethod
    def _references(session: Session, config: RoutingConfig) -> None:
        if session.scalar(select(ModelRecord.id).where(ModelRecord.name == config.model_name)) is None:
            raise ModelNotFoundError("Routing model does not exist")
        found = set(session.scalars(select(ModelVersionRecord.version).where(
            ModelVersionRecord.model_name == config.model_name,
            ModelVersionRecord.version.in_(config.weights),
        )))
        if found != set(config.weights):
            raise ModelVersionNotFoundError("A routing version does not exist")

    @staticmethod
    def _append(session: Session, config: RoutingConfig, revision: int, restored_from: int | None) -> None:
        session.add(RoutingHistoryRecord(model_name=config.model_name, revision=revision,
                                        weights=dict(config.weights), updated_at=config.updated_at,
                                        restored_from=restored_from))
        session.flush()
        session.add_all([RoutingVersionReferenceRecord(model_name=config.model_name, revision=revision, version=version)
                         for version in config.weights])

    def get(self, model_name: str) -> RoutingSnapshot:
        with self._transaction() as session:
            return self._snapshot(self._current(session, model_name))

    def create(self, config: RoutingConfig) -> RoutingSnapshot:
        config = RoutingConfig(config.model_name, config.weights, config.updated_at)
        with self._transaction() as session:
            self._references(session, config)
            session.add(RoutingCurrentRecord(model_name=config.model_name, weights=dict(config.weights),
                                            revision=1, updated_at=config.updated_at, restored_from=None))
            session.flush()
            self._append(session, config, 1, None)
        return RoutingSnapshot(config, 1)

    def _update(self, config: RoutingConfig, expected_revision: int, restored_from: int | None) -> RoutingSnapshot:
        positive_revision(expected_revision)
        config = RoutingConfig(config.model_name, config.weights, config.updated_at)
        with self._transaction() as session:
            # CAS is the first statement, avoiding a read-then-write upgrade race.
            result = session.execute(update(RoutingCurrentRecord).where(
                RoutingCurrentRecord.model_name == config.model_name,
                RoutingCurrentRecord.revision == expected_revision,
            ).values(weights=dict(config.weights), revision=expected_revision + 1,
                     updated_at=config.updated_at, restored_from=restored_from))
            if result.rowcount != 1:
                self._current(session, config.model_name)
                raise RoutingConflictError("Routing changed; re-read before updating")
            self._references(session, config)
            self._append(session, config, expected_revision + 1, restored_from)
        return RoutingSnapshot(config, expected_revision + 1, restored_from)

    def update(self, config: RoutingConfig, *, expected_revision: int) -> RoutingSnapshot:
        return self._update(config, expected_revision, None)

    def history(self, model_name: str) -> list[RoutingSnapshot]:
        with self._transaction() as session:
            self._current(session, model_name)
            return [self._snapshot(row) for row in session.scalars(select(RoutingHistoryRecord).where(
                RoutingHistoryRecord.model_name == model_name,
            ).order_by(RoutingHistoryRecord.revision))]

    def get_revision(self, model_name: str, revision: int) -> RoutingSnapshot:
        positive_revision(revision)
        with self._transaction() as session:
            record = session.get(RoutingHistoryRecord, (model_name, revision))
            if record is None:
                raise RoutingHistoryNotFoundError("Routing history revision does not exist")
            return self._snapshot(record)

    def restore(self, model_name: str, target_revision: int, *, expected_revision: int) -> RoutingSnapshot:
        historical = self.get_revision(model_name, target_revision)
        return self._update(RoutingConfig(model_name, historical.config.weights), expected_revision, target_revision)
