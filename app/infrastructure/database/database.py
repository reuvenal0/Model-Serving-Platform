"""Final SQLite schema: bootstrap empty databases, validate existing ones read-only."""

from functools import lru_cache
import logging
import re

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine, make_url

from app.core.exceptions import RepositoryError
from app.infrastructure.database.records import Base


class SchemaCompatibilityError(RepositoryError):
    """An existing database cannot safely be used with the final schema."""


def create_sqlite_engine(database_url: str) -> Engine:
    if make_url(database_url).get_backend_name() != "sqlite":
        raise ValueError("This adapter requires a SQLite database URL")
    engine = create_engine(database_url)

    @event.listens_for(engine, "connect")
    def configure_connection(connection, connection_record) -> None:
        connection.isolation_level = None
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    @event.listens_for(engine, "begin")
    def begin_transaction(connection) -> None:
        connection.exec_driver_sql("BEGIN")

    return engine


def schema_signature(connection) -> tuple:
    """Compare SQLite's own DDL, including checks, indexes, FKs and triggers.

    A preexisting, unused migration ledger is tolerated but never created, read,
    updated, or removed. No historical layout is upgraded.
    """
    rows = connection.execute(
        text(
            "SELECT type, name, tbl_name, sql FROM sqlite_master "
            "WHERE name NOT GLOB 'sqlite_*' AND tbl_name != 'schema_migrations' "
            "ORDER BY type, name"
        )
    )
    return tuple(
        (kind, name, table, re.sub(r"\s+", " ", sql or "").strip())
        for kind, name, table, sql in rows
    )


@lru_cache(maxsize=1)
def expected_schema() -> tuple:
    # SQLite normalizes CREATE statements on insertion. Generate the reference
    # using the same dialect in private memory, never using caller data.
    reference = create_sqlite_engine("sqlite://")
    try:
        with reference.begin() as connection:
            Base.metadata.create_all(connection)
            return schema_signature(connection)
    finally:
        reference.dispose()


def initialize_database(engine: Engine) -> None:
    """One transactional final bootstrap; existing incompatible schemas fail closed."""
    expected = expected_schema()
    with engine.begin() as connection:
        actual = schema_signature(connection)
        if actual:
            if actual != expected:
                logging.getLogger(__name__).error(
                    "sqlite_schema_incompatible: automatic schema upgrades are unsupported; use a separate empty database"
                )
                raise SchemaCompatibilityError(
                    "Existing SQLite schema is incompatible. Automatic upgrades are unsupported; use a separate empty database."
                )
            return  # No CREATE, DROP, ALTER, or data changes on compatible startup.
        # An old ledger alone is not an empty fresh installation.
        if connection.scalar(
            text("SELECT COUNT(*) FROM sqlite_master WHERE name = 'schema_migrations'")
        ):
            raise SchemaCompatibilityError(
                "Incomplete existing SQLite schema; automatic upgrades are unsupported"
            )
        Base.metadata.create_all(connection)
        if schema_signature(connection) != expected:
            raise SchemaCompatibilityError(
                "Final SQLite schema initialization was not confirmed"
            )
