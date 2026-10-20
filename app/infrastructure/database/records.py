"""Internal SQLAlchemy mappings; these never leave the repository."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, CheckConstraint, Float, ForeignKey, ForeignKeyConstraint, Index, JSON, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator[datetime]):
    """Persist explicit UTC ISO timestamps, avoiding SQLite's timezone loss."""

    impl = String(40)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> str | None:
        if value is None:
            return None
        if value.utcoffset() is None:
            raise ValueError("Timestamp must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat()

    def process_result_value(self, value: str | None, dialect) -> datetime | None:
        if value is None:
            return None
        result = datetime.fromisoformat(value)
        if result.utcoffset() is None:
            raise ValueError("Stored timestamp is missing its UTC offset")
        return result.astimezone(timezone.utc)


class Base(DeclarativeBase):
    pass


class ModelRecord(Base):
    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("name", name="uq_models_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class ModelVersionRecord(Base):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("model_name", "version", name="uq_model_versions_identity"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(
        ForeignKey("models.name", ondelete="RESTRICT"), nullable=False
    )
    version: Mapped[str] = mapped_column(String, nullable=False)
    artifact_location: Mapped[str] = mapped_column(String, nullable=False)
    framework: Mapped[str] = mapped_column(String, nullable=False)
    input_schema: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    output_schema: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class DeploymentRecord(Base):
    __tablename__ = "deployments"
    __table_args__ = (
        ForeignKeyConstraint(
            ["model_name", "version"], ["model_versions.model_name", "model_versions.version"],
            ondelete="RESTRICT", name="fk_deployments_model_version",
        ),
        CheckConstraint("replicas >= 0", name="ck_deployments_replicas"),
        CheckConstraint("cpu > 0 AND memory_mb > 0", name="ck_deployments_resources"),
        CheckConstraint("revision >= 0", name="ck_deployments_revision"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    model_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    version: Mapped[str] = mapped_column(String, nullable=False)
    replicas: Mapped[int] = mapped_column(nullable=False)
    cpu: Mapped[float] = mapped_column(Float, nullable=False)
    memory_mb: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    endpoint: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    revision: Mapped[int] = mapped_column(nullable=False)


class RoutingCurrentRecord(Base):
    __tablename__ = "routing_current"
    __table_args__ = (CheckConstraint("revision >= 1"),)

    model_name: Mapped[str] = mapped_column(ForeignKey("models.name", ondelete="RESTRICT"), primary_key=True)
    weights: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False)
    revision: Mapped[int] = mapped_column(nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    restored_from: Mapped[int | None] = mapped_column(nullable=True)


class RoutingHistoryRecord(Base):
    __tablename__ = "routing_history"
    __table_args__ = (CheckConstraint("revision >= 1"),)

    model_name: Mapped[str] = mapped_column(ForeignKey("models.name", ondelete="RESTRICT"), primary_key=True)
    revision: Mapped[int] = mapped_column(primary_key=True)
    weights: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    restored_from: Mapped[int | None] = mapped_column(nullable=True)


class RoutingVersionReferenceRecord(Base):
    """Normalized historical references enforce artifact-retention safety."""

    __tablename__ = "routing_version_references"
    __table_args__ = (
        ForeignKeyConstraint(["model_name", "revision"],
                             ["routing_history.model_name", "routing_history.revision"], ondelete="RESTRICT"),
        ForeignKeyConstraint(["model_name", "version"],
                             ["model_versions.model_name", "model_versions.version"], ondelete="RESTRICT"),
    )
    model_name: Mapped[str] = mapped_column(String, primary_key=True)
    revision: Mapped[int] = mapped_column(primary_key=True)
    version: Mapped[str] = mapped_column(String, primary_key=True)


class InferenceLogRecord(Base):
    __tablename__ = "inference_logs"
    __table_args__ = (
        CheckConstraint("latency_ms >= 0"),
        CheckConstraint("success IN (0, 1)"),
        CheckConstraint("success = 0 OR (model_name IS NOT NULL AND version IS NOT NULL)"),
        Index("ix_inference_logs_time", "timestamp"),
        Index("ix_inference_logs_model_version_time", "model_name", "version", "timestamp"),
        Index("ix_inference_logs_model_time", "model_name", "timestamp"),
    )
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    model_name: Mapped[str | None] = mapped_column(String, nullable=True)
    version: Mapped[str | None] = mapped_column(String, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    input_hash: Mapped[str | None] = mapped_column(String, nullable=True)
