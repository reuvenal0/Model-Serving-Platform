"""Revision-guarded deployment metadata in SQLite."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
import sqlite3

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.exceptions import (
    DeploymentConflictError, DeploymentNotFoundError, DeploymentPersistenceError,
    DomainValidationError, DuplicateDeploymentError, ModelVersionNotFoundError,
)
from app.domain.deployment import Deployment
from app.domain.enums import DeploymentStatus
from app.domain.resource_config import ResourceConfig
from app.infrastructure.database.records import DeploymentRecord
from app.interfaces.deployment_repository import DeploymentRepository, DeploymentSnapshot


class SQLiteDeploymentRepository(DeploymentRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @contextmanager
    def _transaction(self) -> Iterator[Session]:
        try:
            with self._session_factory.begin() as session:
                yield session
        except IntegrityError:
            raise
        except OperationalError as error:
            code = getattr(error.orig, "sqlite_errorcode", 0) or 0
            if code & 0xff in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise DeploymentConflictError("SQLite is busy; re-read before retrying") from error
            raise DeploymentPersistenceError("Deployment metadata operation failed") from error
        except (SQLAlchemyError, ValueError, TypeError) as error:
            raise DeploymentPersistenceError("Deployment metadata operation failed") from error

    @staticmethod
    def _snapshot(record: DeploymentRecord) -> DeploymentSnapshot:
        return DeploymentSnapshot(Deployment(
            id=record.id, model_name=record.model_name, version=record.version,
            replicas=record.replicas, resources=ResourceConfig(record.cpu, record.memory_mb),
            status=DeploymentStatus(record.status), endpoint=record.endpoint,
            created_at=record.created_at, updated_at=record.updated_at,
        ), record.revision)

    @staticmethod
    def _values(deployment: Deployment) -> dict:
        return dict(
            id=deployment.id, model_name=deployment.model_name, version=deployment.version,
            replicas=deployment.replicas, cpu=deployment.resources.cpu,
            memory_mb=deployment.resources.memory_mb, status=deployment.status.value,
            endpoint=deployment.endpoint, created_at=deployment.created_at,
            updated_at=deployment.updated_at,
        )

    def save_deployment(self, deployment: Deployment) -> DeploymentSnapshot:
        deployment = replace(deployment)
        try:
            with self._transaction() as session:
                session.add(DeploymentRecord(**self._values(deployment), revision=0))
            return DeploymentSnapshot(deployment, 0)
        except IntegrityError as error:
            code = getattr(error.orig, "sqlite_errorcode", None)
            if code in (sqlite3.SQLITE_CONSTRAINT_UNIQUE, sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY):
                raise DuplicateDeploymentError(f"Deployment {deployment.id!r} already exists") from error
            if code == sqlite3.SQLITE_CONSTRAINT_FOREIGNKEY:
                raise ModelVersionNotFoundError(
                    f"Version {deployment.model_name!r}/{deployment.version!r} does not exist"
                ) from error
            raise DeploymentPersistenceError("Could not insert deployment") from error

    def get_deployment(self, deployment_id: str) -> DeploymentSnapshot:
        with self._transaction() as session:
            record = session.get(DeploymentRecord, deployment_id)
            if record is None:
                raise DeploymentNotFoundError(f"Deployment {deployment_id!r} does not exist")
            return self._snapshot(record)

    def list_deployments(self, model_name: str | None = None) -> list[DeploymentSnapshot]:
        with self._transaction() as session:
            statement = select(DeploymentRecord).order_by(DeploymentRecord.id)
            if model_name is not None:
                statement = statement.where(DeploymentRecord.model_name == model_name)
            return [self._snapshot(record) for record in session.scalars(statement)]

    def update_deployment(
        self, deployment: Deployment, *, expected_revision: int,
    ) -> DeploymentSnapshot:
        deployment = replace(deployment)
        if type(expected_revision) is not int or expected_revision < 0:
            raise DomainValidationError("expected_revision must be a nonnegative integer")
        values = self._values(deployment)
        for key in ("id", "model_name", "version", "created_at"):
            del values[key]
        values["revision"] = expected_revision + 1
        try:
            with self._transaction() as session:
                result = session.execute(update(DeploymentRecord).where(
                    DeploymentRecord.id == deployment.id,
                    DeploymentRecord.revision == expected_revision,
                    DeploymentRecord.model_name == deployment.model_name,
                    DeploymentRecord.version == deployment.version,
                    DeploymentRecord.created_at == deployment.created_at,
                ).values(**values))
                if result.rowcount == 0:
                    if session.get(DeploymentRecord, deployment.id) is None:
                        raise DeploymentNotFoundError(f"Deployment {deployment.id!r} does not exist")
                    raise DeploymentConflictError(
                        f"Deployment {deployment.id!r} changed; re-read before retrying"
                    )
            return DeploymentSnapshot(deployment, expected_revision + 1)
        except IntegrityError as error:
            raise DeploymentPersistenceError("Could not update deployment") from error
