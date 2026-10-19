import pytest
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

from app.infrastructure.database.database import create_sqlite_engine, initialize_database
from app.infrastructure.database.sqlite_model_repository import SQLiteModelRepository


@pytest.fixture
def database_url(tmp_path):
    return URL.create("sqlite", database=str(tmp_path / "test.db")).render_as_string()


@pytest.fixture
def engine(database_url):
    engine = create_sqlite_engine(database_url)
    initialize_database(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def repository(engine):
    return SQLiteModelRepository(sessionmaker(engine))
