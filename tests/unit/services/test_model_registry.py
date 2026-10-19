from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.core.exceptions import (
    ArtifactCleanupError, ArtifactNotFoundError, ArtifactStorageError,
    DomainValidationError, DuplicateArtifactError, DuplicateModelError,
    DuplicateVersionError, ModelNotFoundError, ModelVersionNotFoundError,
    RegistryConsistencyError, RepositoryError,
)
from app.domain.enums import ModelFramework
from app.domain.model import Model
from app.domain.model_version import ModelVersion
from app.interfaces.artifact_storage import ArtifactStorage
from app.interfaces.model_repository import ModelRepository
from app.services.model_registry import ModelRegistry


@pytest.fixture
def dependencies():
    repository = Mock(spec=ModelRepository)
    storage = Mock(spec=ArtifactStorage)
    repository.get_model.return_value = Model("classifier")
    repository.get_version.side_effect = ModelVersionNotFoundError("missing")
    registry = ModelRegistry(repository, storage)
    return SimpleNamespace(repository=repository, storage=storage, registry=registry)


def register(dependencies, **overrides):
    values = dict(model_name="classifier", version="v1", framework=ModelFramework.ONNX,
                  content=b"harmless data")
    return dependencies.registry.register_version(**(values | overrides))


def existing_version(dependencies):
    item = ModelVersion("classifier", "v1", "artifacts/existing", ModelFramework.ONNX)
    dependencies.repository.get_version.side_effect = None
    dependencies.repository.get_version.return_value = item
    dependencies.repository.delete_version.return_value = item
    return item


def test_register_model(dependencies):
    dependencies.repository.get_model.side_effect = ModelNotFoundError("missing")
    result = dependencies.registry.register_model("classifier", "A model")
    assert result.name == "classifier"
    assert result.description == "A model"
    dependencies.repository.save_model.assert_called_once_with(result)


def test_duplicate_model(dependencies):
    with pytest.raises(DuplicateModelError):
        dependencies.registry.register_model("classifier")
    dependencies.repository.save_model.assert_not_called()


def test_model_lookup_backend_failure_does_not_trigger_save(dependencies):
    dependencies.repository.get_model.side_effect = RepositoryError("unavailable")
    with pytest.raises(RepositoryError):
        dependencies.registry.register_model("classifier")
    dependencies.repository.save_model.assert_not_called()


def test_register_version_uses_interfaces_in_order(dependencies):
    calls = Mock()
    calls.attach_mock(dependencies.repository, "repository")
    calls.attach_mock(dependencies.storage, "storage")
    item = register(dependencies, input_schema={"features": {"type": "array"}})
    assert item.model_name == "classifier"
    assert item.version == "v1"
    assert item.input_schema == {"features": {"type": "array"}}
    assert item.artifact_location.startswith("artifacts/")
    dependencies.storage.save.assert_called_once_with(item.artifact_location, b"harmless data")
    dependencies.repository.save_version.assert_called_once_with(item)
    dependencies.storage.delete.assert_not_called()
    assert [call[0] for call in calls.mock_calls] == [
        "repository.get_model", "repository.get_version", "storage.save", "repository.save_version",
    ]


def test_keys_are_unique_and_independent_of_identifiers(dependencies):
    first = register(dependencies, model_name="../opaque-model", version="../../v1")
    second = register(dependencies, version="v2")
    assert first.artifact_location != second.artifact_location
    assert ".." not in first.artifact_location
    assert len(first.artifact_location.split("/")[1]) == 32


def test_missing_parent_prevents_artifact_creation(dependencies):
    dependencies.repository.get_model.side_effect = ModelNotFoundError("missing")
    with pytest.raises(ModelNotFoundError):
        register(dependencies)
    dependencies.storage.save.assert_not_called()
    dependencies.repository.save_version.assert_not_called()


def test_duplicate_version_does_not_touch_storage(dependencies):
    existing_version(dependencies)
    with pytest.raises(DuplicateVersionError):
        register(dependencies)
    dependencies.storage.save.assert_not_called()
    dependencies.storage.delete.assert_not_called()
    dependencies.repository.save_version.assert_not_called()


def test_version_lookup_backend_failure_does_not_touch_storage(dependencies):
    dependencies.repository.get_version.side_effect = RepositoryError("unavailable")
    with pytest.raises(RepositoryError):
        register(dependencies)
    dependencies.storage.save.assert_not_called()


@pytest.mark.parametrize("overrides", [
    {"model_name": " "}, {"version": ""}, {"framework": "onnx"},
    {"content": "text"}, {"input_schema": []}, {"output_schema": {"x": object()}},
    {"input_schema": {"x": float("nan")}},
])
def test_invalid_registration_fails_before_io(dependencies, overrides):
    with pytest.raises(DomainValidationError):
        register(dependencies, **overrides)
    dependencies.repository.get_model.assert_not_called()
    dependencies.storage.save.assert_not_called()


def test_schemas_are_copied(dependencies):
    schema = {"features": ["x"]}
    item = register(dependencies, input_schema=schema)
    schema["features"].append("y")
    assert item.input_schema == {"features": ["x"]}


def test_metadata_failure_cleans_only_created_artifact(dependencies):
    failure = RepositoryError("database failed")
    dependencies.repository.save_version.side_effect = failure
    with pytest.raises(RepositoryError) as captured:
        register(dependencies)
    assert captured.value is failure
    key = dependencies.storage.save.call_args.args[0]
    dependencies.storage.delete.assert_called_once_with(key)


def test_duplicate_race_also_cleans_new_artifact(dependencies):
    dependencies.repository.save_version.side_effect = DuplicateVersionError("concurrent insert")
    with pytest.raises(DuplicateVersionError):
        register(dependencies)
    key = dependencies.storage.save.call_args.args[0]
    dependencies.storage.delete.assert_called_once_with(key)


def test_artifact_collision_never_deletes_preexisting_bytes(dependencies):
    dependencies.storage.save.side_effect = DuplicateArtifactError("key exists")
    with patch("app.services.model_registry.uuid4", return_value=SimpleNamespace(hex="collision")):
        with pytest.raises(DuplicateArtifactError):
            register(dependencies)
    dependencies.storage.save.assert_called_once_with("artifacts/collision", b"harmless data")
    dependencies.storage.delete.assert_not_called()
    dependencies.repository.save_version.assert_not_called()


def test_storage_failure_does_not_create_metadata_or_delete_anything(dependencies):
    dependencies.storage.save.side_effect = ArtifactStorageError("disk full")
    with pytest.raises(ArtifactStorageError):
        register(dependencies)
    dependencies.repository.save_version.assert_not_called()
    dependencies.storage.delete.assert_not_called()


def test_artifact_temporary_cleanup_failure_is_explicit(dependencies):
    failure = ArtifactCleanupError("key", "temporary-file", published=True)
    dependencies.storage.save.side_effect = failure
    with pytest.raises(RegistryConsistencyError) as captured:
        register(dependencies)
    assert captured.value.operation_error is failure
    dependencies.repository.save_version.assert_not_called()
    dependencies.storage.delete.assert_not_called()


def test_failed_rollback_reports_both_errors_and_orphan_key(dependencies):
    operation = RepositoryError("database failed")
    cleanup = ArtifactStorageError("cannot delete")
    dependencies.repository.save_version.side_effect = operation
    dependencies.storage.delete.side_effect = cleanup
    with pytest.raises(RegistryConsistencyError) as captured:
        register(dependencies)
    error = captured.value
    assert error.operation_error is operation
    assert error.cleanup_error is cleanup
    assert error.artifact_key == dependencies.storage.save.call_args.args[0]
    assert error.__cause__ is cleanup


def test_download_artifact_uses_stored_key(dependencies):
    item = existing_version(dependencies)
    dependencies.storage.retrieve.return_value = b"artifact"
    assert dependencies.registry.download_artifact("classifier", "v1") == b"artifact"
    dependencies.storage.retrieve.assert_called_once_with(item.artifact_location)


def test_missing_version_download_does_not_access_storage(dependencies):
    with pytest.raises(ModelVersionNotFoundError):
        dependencies.registry.download_artifact("classifier", "missing")
    dependencies.storage.retrieve.assert_not_called()


def test_missing_artifact_download_reports_storage_error(dependencies):
    existing_version(dependencies)
    dependencies.storage.retrieve.side_effect = ArtifactNotFoundError("missing bytes")
    with pytest.raises(ArtifactNotFoundError):
        dependencies.registry.download_artifact("classifier", "v1")


def test_delete_commits_metadata_before_artifact(dependencies):
    item = existing_version(dependencies)
    calls = Mock()
    calls.attach_mock(dependencies.repository, "repository")
    calls.attach_mock(dependencies.storage, "storage")
    assert dependencies.registry.delete_version("classifier", "v1") is None
    assert [call[0] for call in calls.mock_calls] == [
        "repository.get_version", "repository.delete_version", "storage.delete",
    ]
    dependencies.storage.delete.assert_called_once_with(item.artifact_location)


def test_missing_version_deletion_touches_neither_backend(dependencies):
    with pytest.raises(ModelVersionNotFoundError):
        dependencies.registry.delete_version("classifier", "v1")
    dependencies.repository.delete_version.assert_not_called()
    dependencies.storage.delete.assert_not_called()


def test_metadata_deletion_failure_preserves_artifact(dependencies):
    existing_version(dependencies)
    dependencies.repository.delete_version.side_effect = RepositoryError("database failed")
    with pytest.raises(RepositoryError):
        dependencies.registry.delete_version("classifier", "v1")
    dependencies.storage.delete.assert_not_called()


def test_artifact_deletion_failure_reports_partial_state(dependencies):
    item = existing_version(dependencies)
    failure = ArtifactStorageError("cannot delete")
    dependencies.storage.delete.side_effect = failure
    with pytest.raises(RegistryConsistencyError) as captured:
        dependencies.registry.delete_version("classifier", "v1")
    dependencies.repository.delete_version.assert_called_once_with("classifier", "v1")
    assert captured.value.artifact_key == item.artifact_location
    assert captured.value.cleanup_error is failure
    assert captured.value.operation_error is None


def test_delete_uses_the_snapshot_actually_removed(dependencies):
    existing_version(dependencies)
    removed = ModelVersion("classifier", "v1", "artifacts/actual", ModelFramework.ONNX)
    dependencies.repository.delete_version.return_value = removed
    dependencies.registry.delete_version("classifier", "v1")
    dependencies.storage.delete.assert_called_once_with("artifacts/actual")
