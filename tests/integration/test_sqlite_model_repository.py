from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, insert, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.core.exceptions import (
    DomainValidationError, DuplicateModelError, DuplicateVersionError,
    ModelNotFoundError, ModelVersionNotFoundError, RepositoryError,
)
from app.domain.enums import ModelFramework, ModelVersionStatus
from app.domain.model import Model
from app.domain.model_version import ModelVersion
from app.infrastructure.database.database import create_sqlite_engine, initialize_database
from app.infrastructure.database.records import ModelRecord, ModelVersionRecord
from app.infrastructure.database.sqlite_model_repository import SQLiteModelRepository


def version(model_name="classifier", version="v1", **overrides):
    values = dict(model_name=model_name, version=version, artifact_location="artifacts/key",
                  framework=ModelFramework.SKLEARN)
    return ModelVersion(**(values | overrides))


def test_save_and_get_model(repository):
    model = Model("classifier", "Test classifier")
    assert repository.save_model(model) is None
    assert repository.get_model("classifier") == model
    assert repository.get_model("classifier") is not model


def test_missing_model(repository):
    with pytest.raises(ModelNotFoundError):
        repository.get_model("missing")


def test_duplicate_model_preserves_original_and_session_recovers(repository):
    repository.save_model(Model("classifier", "Original"))
    with pytest.raises(DuplicateModelError):
        repository.save_model(Model("classifier", "Replacement"))
    repository.save_model(Model("another"))
    assert repository.get_model("classifier").description == "Original"
    assert len(repository.list_models()) == 2


def test_list_models_is_sorted(repository):
    assert repository.list_models() == []
    for name in ["z", "a", "m"]:
        repository.save_model(Model(name))
    assert [model.name for model in repository.list_models()] == ["a", "m", "z"]


def test_save_get_and_list_versions(repository):
    repository.save_model(Model("classifier"))
    assert repository.list_versions("classifier") == []
    for identifier in ["v2", "v1"]:
        item = version(version=identifier, input_schema={"features": ["x"]},
                       output_schema={"prediction": {"type": "number"}},
                       status=ModelVersionStatus.INVALID)
        assert repository.save_version(item) is None
        assert repository.get_version("classifier", identifier) == item
    assert [v.version for v in repository.list_versions("classifier")] == ["v1", "v2"]
    first = repository.get_version("classifier", "v1")
    first.input_schema["features"].append("y")
    assert repository.get_version("classifier", "v1").input_schema == {"features": ["x"]}


def test_same_version_identifier_under_different_models(repository):
    for name in ["one", "two"]:
        repository.save_model(Model(name))
        repository.save_version(version(model_name=name))
    assert repository.get_version("one", "v1").model_name == "one"
    assert repository.get_version("two", "v1").model_name == "two"


def test_duplicate_version(repository):
    repository.save_model(Model("classifier"))
    original = version()
    repository.save_version(original)
    with pytest.raises(DuplicateVersionError):
        repository.save_version(version(artifact_location="replacement"))
    assert repository.get_version("classifier", "v1") == original


def test_version_requires_parent(repository):
    with pytest.raises(ModelNotFoundError):
        repository.save_version(version())
    with pytest.raises(ModelNotFoundError):
        repository.list_versions("missing")
    with pytest.raises(ModelVersionNotFoundError):
        repository.get_version("missing", "v1")


def test_delete_version_preserves_parent_and_other_versions(repository):
    repository.save_model(Model("classifier"))
    repository.save_version(version())
    repository.save_version(version(version="v2"))
    removed = repository.delete_version("classifier", "v1")
    assert removed.model_name == "classifier"
    assert removed.version == "v1"
    with pytest.raises(ModelVersionNotFoundError):
        repository.get_version("classifier", "v1")
    with pytest.raises(ModelVersionNotFoundError):
        repository.delete_version("classifier", "v1")
    assert repository.get_model("classifier").name == "classifier"
    assert [v.version for v in repository.list_versions("classifier")] == ["v2"]


def test_persistence_across_engines(database_url):
    first_engine = create_sqlite_engine(database_url)
    try:
        initialize_database(first_engine)
        first = SQLiteModelRepository(sessionmaker(first_engine))
        first.save_model(Model("classifier"))
        item = version()
        first.save_version(item)
    finally:
        first_engine.dispose()
    second_engine = create_sqlite_engine(database_url)
    try:
        second = SQLiteModelRepository(sessionmaker(second_engine))
        assert second.get_model("classifier").name == "classifier"
        assert second.get_version("classifier", "v1") == item
        initialize_database(second_engine)  # Initialization is repeatable.
        assert second.get_version("classifier", "v1") == item
    finally:
        second_engine.dispose()


def test_foreign_keys_enabled_on_each_connection(engine):
    with engine.connect() as first, engine.connect() as second:
        assert first.scalar(text("PRAGMA foreign_keys")) == 1
        assert second.scalar(text("PRAGMA foreign_keys")) == 1


def test_database_rejects_orphan_version_even_without_repository(engine):
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(insert(ModelVersionRecord).values(
                model_name="missing", version="v1", artifact_location="key",
                framework="onnx", input_schema={}, output_schema={},
                status="registered", created_at=datetime.now(timezone.utc),
            ))


def test_database_prevents_parent_deletion(repository, engine):
    repository.save_model(Model("classifier"))
    repository.save_version(version())
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(delete(ModelRecord).where(ModelRecord.name == "classifier"))
    assert repository.get_version("classifier", "v1").version == "v1"


def test_timestamps_round_trip_without_losing_offset_or_microseconds(repository):
    local = datetime(2026, 1, 1, 12, 30, 4, 123456,
                     tzinfo=timezone(timedelta(hours=5, minutes=30)))
    model = Model("classifier", created_at=local, updated_at=local + timedelta(seconds=7))
    item = version(created_at=local)
    repository.save_model(model)
    repository.save_version(item)
    retrieved = repository.get_model("classifier")
    retrieved_version = repository.get_version("classifier", "v1")
    assert retrieved == model
    assert retrieved_version == item
    for timestamp in [retrieved.created_at, retrieved.updated_at, retrieved_version.created_at]:
        assert timestamp.tzinfo is timezone.utc
        assert timestamp.microsecond == 123456


def test_mutated_naive_timestamps_are_rejected(repository):
    model = Model("classifier")
    model.created_at = datetime(2026, 1, 1)
    with pytest.raises(DomainValidationError):
        repository.save_model(model)
    assert repository.list_models() == []
    repository.save_model(Model("classifier"))
    item = version()
    item.created_at = datetime(2026, 1, 1)
    with pytest.raises(DomainValidationError):
        repository.save_version(item)
    assert repository.list_versions("classifier") == []


def test_failed_json_write_rolls_back(repository):
    repository.save_model(Model("classifier"))
    with pytest.raises(RepositoryError):
        repository.save_version(version(input_schema={"invalid": object()}))
    assert repository.list_versions("classifier") == []
    repository.save_version(version())
    assert repository.get_version("classifier", "v1").version == "v1"


def test_non_duplicate_integrity_error_is_not_mislabeled(repository):
    with pytest.raises(RepositoryError) as captured:
        repository.save_model(Model("classifier", description=None))
    assert not isinstance(captured.value, DuplicateModelError)
    assert repository.list_models() == []


def test_backend_failure_is_translated(engine):
    engine_without_schema = create_sqlite_engine("sqlite:///:memory:")
    try:
        repository = SQLiteModelRepository(sessionmaker(engine_without_schema))
        with pytest.raises(RepositoryError):
            repository.list_models()
    finally:
        engine_without_schema.dispose()


@pytest.mark.parametrize("field,value", [
    ("created_at", "2026-01-01T12:00:00"),
    ("created_at", "not-a-timestamp"),
    ("framework", "unsupported-framework"),
])
def test_corrupt_stored_metadata_raises_repository_error(repository, engine, field, value):
    repository.save_model(Model("classifier"))
    repository.save_version(version())
    # Field names are fixed test parameters; bypass ORM conversion deliberately.
    with engine.begin() as connection:
        connection.execute(text(
            f"UPDATE model_versions SET {field} = :value WHERE model_name = :name"
        ), {"value": value, "name": "classifier"})
    with pytest.raises(RepositoryError):
        repository.get_version("classifier", "v1")
