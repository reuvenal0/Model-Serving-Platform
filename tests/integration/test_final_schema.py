"""The final bootstrap is complete, idempotent, and never upgrades existing data."""

from pathlib import Path
import sqlite3

import pytest
from sqlalchemy import event, inspect, text

from app.infrastructure.database.database import (
    create_sqlite_engine,
    initialize_database,
    schema_signature,
    SchemaCompatibilityError,
)
from app.infrastructure.database.records import Base

REFERENCE = Path(__file__).parents[1] / "fixtures" / "final_schema.sql"


def test_final_metadata_matches_captured_historical_complete_schema(tmp_path):
    # The fixture is immutable old DDL, not executable migration machinery.
    statements = []
    current = ""
    for line in REFERENCE.read_text().splitlines(keepends=True):
        current += line
        if sqlite3.complete_statement(current):
            statements.append(current)
            current = ""
    old = sqlite3.connect(tmp_path / "reference.db")
    try:
        for statement in sorted(
            statements, key=lambda s: 0 if "CREATE TABLE" in s else 1
        ):
            old.executescript(statement)
    finally:
        old.close()
    reference = create_sqlite_engine(f"sqlite:///{tmp_path / 'reference.db'}")
    final = create_sqlite_engine(f"sqlite:///{tmp_path / 'final.db'}")
    try:
        initialize_database(final)
        with reference.connect() as first, final.connect() as second:
            assert schema_signature(first) == schema_signature(second)
        assert set(inspect(final).get_table_names()) == set(Base.metadata.tables)
        assert "schema_migrations" not in inspect(final).get_table_names()
    finally:
        reference.dispose()
        final.dispose()


def test_compatible_restart_issues_no_ddl_or_data_writes(tmp_path):
    path = tmp_path / "data.db"
    engine = create_sqlite_engine(f"sqlite:///{path}")
    initialize_database(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO models(name,description,created_at,updated_at) VALUES('saved','data','2026-01-01T00:00:00+00:00','2026-01-01T00:00:00+00:00')"
            )
        )
        # A legacy ledger may remain but is never changed or recreated.
        connection.execute(
            text("CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY)")
        )
        connection.execute(text("INSERT INTO schema_migrations VALUES(6),(7)"))
    before = path.read_bytes()
    statements = []
    event.listen(
        engine,
        "before_cursor_execute",
        lambda c, cur, s, p, ctx, m: statements.append(s),
    )
    initialize_database(engine)
    initialize_database(engine)
    assert path.read_bytes() == before
    assert not any(
        s.lstrip()
        .upper()
        .startswith(("CREATE", "DROP", "ALTER", "INSERT", "UPDATE", "DELETE"))
        for s in statements
    )
    engine.dispose()


@pytest.mark.parametrize(
    "change",
    [
        "DROP TABLE inference_logs",
        "DROP INDEX ix_inference_logs_time",
        "DROP TRIGGER routing_history_no_update",
        "ALTER TABLE models ADD COLUMN incompatible TEXT",
        "CREATE TABLE unknown_data(id INTEGER)",
    ],
)
def test_incompatible_schema_fails_without_partial_upgrade(tmp_path, change):
    path = tmp_path / "data.db"
    engine = create_sqlite_engine(f"sqlite:///{path}")
    initialize_database(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql(change)
    before = path.read_bytes()
    with pytest.raises(SchemaCompatibilityError, match="unsupported"):
        initialize_database(engine)
    assert path.read_bytes() == before
    engine.dispose()


def test_legacy_partial_schema_is_rejected_without_adding_tables(tmp_path):
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'partial.db'}")
    with engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE models(id INTEGER PRIMARY KEY, name TEXT)")
        )
    with pytest.raises(SchemaCompatibilityError):
        initialize_database(engine)
    assert inspect(engine).get_table_names() == ["models"]
    engine.dispose()


def test_failed_fresh_bootstrap_rolls_back_all_ddl_and_retries(tmp_path):
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'fresh.db'}")

    def fail(conn, cursor, statement, parameters, context, executemany):
        if "CREATE INDEX ix_inference_logs_time" in statement:
            raise RuntimeError("controlled failure")

    event.listen(engine, "before_cursor_execute", fail)
    with pytest.raises(RuntimeError):
        initialize_database(engine)
    event.remove(engine, "before_cursor_execute", fail)
    assert inspect(engine).get_table_names() == []
    initialize_database(engine)
    assert len(inspect(engine).get_table_names()) == 7
    with engine.connect() as connection:
        assert connection.scalar(text("PRAGMA foreign_keys")) == 1
    engine.dispose()


def test_unknown_sqlite_lookalike_table_is_not_treated_as_empty(tmp_path):
    path = tmp_path / "unrecognized.db"
    engine = create_sqlite_engine(f"sqlite:///{path}")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE sqliteXlegacy(value TEXT)")
        connection.exec_driver_sql("INSERT INTO sqliteXlegacy VALUES('preserve')")
    before = path.read_bytes()
    with pytest.raises(SchemaCompatibilityError):
        initialize_database(engine)
    assert path.read_bytes() == before
    assert inspect(engine).get_table_names() == ["sqliteXlegacy"]
    engine.dispose()
