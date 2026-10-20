"""Versioned additive migrations for existing SQLite databases.

No legacy table is rebuilt. Migration DDL and its ledger entry commit together.
Run initialization before accepting traffic; concurrent startup can report busy
and must be retried by the operator rather than masking a partial migration.
"""

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.infrastructure.database.records import (
    InferenceLogRecord, RoutingCurrentRecord, RoutingHistoryRecord, RoutingVersionReferenceRecord,
)


def migrate_routing(engine: Engine) -> None:
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)"))
        if connection.scalar(text("SELECT version FROM schema_migrations WHERE version = 6")) is not None:
            return
        # This numbered migration owns these new tables; unexpected existing
        # tables without the ledger entry fail instead of being silently adopted.
        for record in (RoutingCurrentRecord, RoutingHistoryRecord, RoutingVersionReferenceRecord):
            record.__table__.create(connection)
        # Protect audit history even against accidental direct SQL updates.
        for table in ("routing_history", "routing_version_references"):
            for operation in ("UPDATE", "DELETE"):
                connection.execute(text(
                    f"CREATE TRIGGER {table}_no_{operation.lower()} BEFORE {operation} ON {table} "
                    "BEGIN SELECT RAISE(ABORT, 'Routing history is immutable'); END"
                ))
        connection.execute(text("INSERT INTO schema_migrations (version) VALUES (6)"))


def migrate_monitoring(engine: Engine) -> None:
    """Migration 7: additive observation table/indexes; existing data untouched."""
    with engine.begin() as connection:
        if connection.scalar(text("SELECT version FROM schema_migrations WHERE version = 7")) is not None:
            return
        InferenceLogRecord.__table__.create(connection)
        connection.execute(text("INSERT INTO schema_migrations (version) VALUES (7)"))
