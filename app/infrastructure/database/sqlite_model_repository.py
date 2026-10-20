"""SQLite metadata repository with one transaction per operation."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
import sqlite3

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.exceptions import (
    DuplicateModelError, DuplicateVersionError, ModelNotFoundError,
    ModelVersionNotFoundError, ModelVersionInUseError, RepositoryError,
)
from app.domain.enums import ModelFramework, ModelVersionStatus
from app.domain.model import Model
from app.domain.model_version import ModelVersion
from app.infrastructure.database.records import ModelRecord, ModelVersionRecord
from app.interfaces.model_repository import ModelRepository


class SQLiteModelRepository(ModelRepository):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    @contextmanager
    def _transaction(self) -> Iterator[Session]:
        try:
            with self._session_factory.begin() as session:
                yield session
        except IntegrityError:
            # Writes below translate specific constraint violations after rollback.
            raise
        except (SQLAlchemyError, ValueError, TypeError) as error:
            raise RepositoryError("Metadata operation failed") from error

    @staticmethod
    def _model(record: ModelRecord) -> Model:
        return Model(record.name, record.description, record.created_at, record.updated_at)

    @staticmethod
    def _version(record: ModelVersionRecord) -> ModelVersion:
        return ModelVersion(
            model_name=record.model_name,
            version=record.version,
            artifact_location=record.artifact_location,
            framework=ModelFramework(record.framework),
            input_schema=record.input_schema,
            output_schema=record.output_schema,
            status=ModelVersionStatus(record.status),
            created_at=record.created_at,
        )

    @staticmethod
    def _require_model(session: Session, name: str) -> ModelRecord:
        record = session.scalar(select(ModelRecord).where(ModelRecord.name == name))
        if record is None:
            raise ModelNotFoundError(f"Model {name!r} does not exist")
        return record

    def save_model(self, model: Model) -> None:
        model = replace(model)  # Revalidate mutable domain snapshots before writing.
        try:
            with self._transaction() as session:
                session.add(ModelRecord(
                    name=model.name, description=model.description,
                    created_at=model.created_at, updated_at=model.updated_at,
                ))
        except IntegrityError as error:
            if getattr(error.orig, "sqlite_errorcode", None) == sqlite3.SQLITE_CONSTRAINT_UNIQUE:
                raise DuplicateModelError(f"Model {model.name!r} already exists") from error
            raise RepositoryError("Could not save model") from error

    def get_model(self, name: str) -> Model:
        with self._transaction() as session:
            return self._model(self._require_model(session, name))

    def list_models(self) -> list[Model]:
        with self._transaction() as session:
            return [self._model(row) for row in session.scalars(
                select(ModelRecord).order_by(ModelRecord.name)
            )]

    def save_version(self, version: ModelVersion) -> None:
        version = replace(version)
        try:
            with self._transaction() as session:
                self._require_model(session, version.model_name)
                session.add(ModelVersionRecord(
                    model_name=version.model_name, version=version.version,
                    artifact_location=version.artifact_location,
                    framework=version.framework.value,
                    input_schema=version.input_schema, output_schema=version.output_schema,
                    status=version.status.value, created_at=version.created_at,
                ))
        except IntegrityError as error:
            code = getattr(error.orig, "sqlite_errorcode", None)
            if code == sqlite3.SQLITE_CONSTRAINT_UNIQUE:
                raise DuplicateVersionError(
                    f"Version {version.model_name!r}/{version.version!r} already exists"
                ) from error
            if code == sqlite3.SQLITE_CONSTRAINT_FOREIGNKEY:
                raise ModelNotFoundError(f"Model {version.model_name!r} does not exist") from error
            raise RepositoryError("Could not save version") from error

    def get_version(self, model_name: str, version: str) -> ModelVersion:
        with self._transaction() as session:
            record = session.scalar(select(ModelVersionRecord).where(
                ModelVersionRecord.model_name == model_name,
                ModelVersionRecord.version == version,
            ))
            if record is None:
                raise ModelVersionNotFoundError(f"Version {model_name!r}/{version!r} does not exist")
            return self._version(record)

    def list_versions(self, model_name: str) -> list[ModelVersion]:
        with self._transaction() as session:
            self._require_model(session, model_name)
            return [self._version(row) for row in session.scalars(
                select(ModelVersionRecord).where(ModelVersionRecord.model_name == model_name)
                .order_by(ModelVersionRecord.version)
            )]

    def delete_version(self, model_name: str, version: str) -> ModelVersion:
        try:
            with self._transaction() as session:
                record = session.scalar(select(ModelVersionRecord).where(
                    ModelVersionRecord.model_name == model_name,
                    ModelVersionRecord.version == version,
                ))
                if record is None:
                    raise ModelVersionNotFoundError(f"Version {model_name!r}/{version!r} does not exist")
                deleted = self._version(record)
                session.delete(record)
                # The context commits before the caller receives the deleted snapshot.
                return deleted
        except IntegrityError as error:
            # SQLite RESTRICT can use CONSTRAINT_TRIGGER rather than FOREIGNKEY.
            if "FOREIGN KEY constraint failed" in str(error.orig):
                raise ModelVersionInUseError(
                    f"Version {model_name!r}/{version!r} is referenced by deployment or routing history"
                ) from error
            raise RepositoryError("Could not delete version") from error
