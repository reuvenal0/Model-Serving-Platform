"""Explicit SQLite engine and schema initialization, with no global session."""

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine, make_url

from app.infrastructure.database.records import Base


def create_sqlite_engine(database_url: str) -> Engine:
    """Create a lazy engine; its owner must dispose it when finished."""
    if make_url(database_url).get_backend_name() != "sqlite":
        raise ValueError("This adapter requires a SQLite database URL")
    engine = create_engine(database_url)

    @event.listens_for(engine, "connect")
    def configure_connection(connection, connection_record) -> None:
        # Let SQLAlchemy begin transactions consistently across Python versions.
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


def initialize_database(engine: Engine) -> None:
    """Create missing tables explicitly; this is not a schema migration system."""
    Base.metadata.create_all(engine)
