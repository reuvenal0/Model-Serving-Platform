import pytest
from sqlalchemy.orm import sessionmaker

from app.core.exceptions import (
    ArtifactStorageError, DuplicateArtifactError, DuplicateModelError,
    DuplicateVersionError, ModelNotFoundError, RepositoryError,
    ModelVersionNotFoundError, RegistryConsistencyError,
)
from app.domain.enums import ModelFramework
from app.infrastructure.database.sqlite_model_repository import SQLiteModelRepository
from app.infrastructure.storage.local_artifact_storage import LocalArtifactStorage
from app.services.model_registry import ModelRegistry


@pytest.fixture
def registry(repository, tmp_path):
    return ModelRegistry(repository, LocalArtifactStorage(tmp_path / "artifacts"))


def test_complete_registry_workflow(registry, engine, tmp_path):
    model = registry.register_model("classifier", "A classifier")
    content = b"harmless artifact\x00\xff"
    registered = registry.register_version(
        "classifier", "v1", ModelFramework.ONNX, content,
        input_schema={"type": "array"}, output_schema={"type": "number"},
    )
    assert registry.get_model("classifier") == model
    assert registry.list_models() == [model]
    assert registry.list_versions("classifier") == [registered]
    assert registry.get_version("classifier", "v1") == registered
    assert registry.download_artifact("classifier", "v1") == content
    second = ModelRegistry(SQLiteModelRepository(sessionmaker(engine)),
                           LocalArtifactStorage(tmp_path / "artifacts"))
    assert second.download_artifact("classifier", "v1") == content
    registry.delete_version("classifier", "v1")
    assert registry.list_versions("classifier") == []
    assert registry.get_model("classifier") == model
    assert list((tmp_path / "artifacts").iterdir()) == []
    with pytest.raises(ModelVersionNotFoundError):
        registry.get_version("classifier", "v1")


def test_duplicates_preserve_registered_data(registry):
    registry.register_model("classifier", "original")
    with pytest.raises(DuplicateModelError):
        registry.register_model("classifier", "replacement")
    first = registry.register_version("classifier", "v1", ModelFramework.SKLEARN, b"original")
    with pytest.raises(DuplicateVersionError):
        registry.register_version("classifier", "v1", ModelFramework.ONNX, b"replacement")
    assert registry.get_model("classifier").description == "original"
    assert registry.get_version("classifier", "v1") == first
    assert registry.download_artifact("classifier", "v1") == b"original"


def test_missing_parent_does_not_create_files(registry, tmp_path):
    with pytest.raises(ModelNotFoundError):
        registry.register_version("missing", "v1", ModelFramework.SKLEARN, b"data")
    assert list((tmp_path / "artifacts").iterdir()) == []


def test_partial_deletion_reports_key_for_direct_cleanup(registry, tmp_path):
    registry.register_model("classifier")
    item = registry.register_version("classifier", "v1", ModelFramework.ONNX, b"data")
    storage = LocalArtifactStorage(tmp_path / "artifacts")
    storage.delete(item.artifact_location)
    with pytest.raises(RegistryConsistencyError) as captured:
        registry.delete_version("classifier", "v1")
    assert captured.value.artifact_key == item.artifact_location
    assert registry.list_versions("classifier") == []
    assert registry.get_model("classifier").name == "classifier"


def test_metadata_failure_removes_actual_new_file(registry, repository, tmp_path, monkeypatch):
    registry.register_model("classifier")
    failure = RepositoryError("simulated database failure")

    def fail_save(version):
        raise failure

    monkeypatch.setattr(repository, "save_version", fail_save)
    with pytest.raises(RepositoryError) as captured:
        registry.register_version("classifier", "v1", ModelFramework.ONNX, b"data")
    assert captured.value is failure
    assert repository.list_versions("classifier") == []
    assert list((tmp_path / "artifacts").iterdir()) == []


def test_collision_preserves_actual_preexisting_artifact(registry, repository, tmp_path, monkeypatch):
    from types import SimpleNamespace

    registry.register_model("classifier")
    storage = LocalArtifactStorage(tmp_path / "artifacts")
    storage.save("artifacts/collision", b"original")
    monkeypatch.setattr("app.services.model_registry.uuid4", lambda: SimpleNamespace(hex="collision"))
    with pytest.raises(DuplicateArtifactError):
        registry.register_version("classifier", "v1", ModelFramework.ONNX, b"replacement")
    assert storage.retrieve("artifacts/collision") == b"original"
    assert repository.list_versions("classifier") == []
    assert len(list((tmp_path / "artifacts").iterdir())) == 1


def test_failed_rollback_leaves_reported_orphan_for_explicit_cleanup(
    registry, repository, tmp_path, monkeypatch,
):
    registry.register_model("classifier")
    storage = LocalArtifactStorage(tmp_path / "artifacts")

    def fail_save(version):
        raise RepositoryError("simulated database failure")

    def fail_delete(self, key):
        raise ArtifactStorageError("simulated storage deletion failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(repository, "save_version", fail_save)
        scoped.setattr(LocalArtifactStorage, "delete", fail_delete)
        with pytest.raises(RegistryConsistencyError) as captured:
            registry.register_version("classifier", "v1", ModelFramework.ONNX, b"orphan")
    key = captured.value.artifact_key
    assert repository.list_versions("classifier") == []
    assert storage.retrieve(key) == b"orphan"
    storage.delete(key)
    assert not storage.exists(key)


def test_failed_artifact_deletion_can_be_reconciled_directly(registry, tmp_path, monkeypatch):
    registry.register_model("classifier")
    item = registry.register_version("classifier", "v1", ModelFramework.ONNX, b"orphan")

    def fail_delete(self, key):
        raise ArtifactStorageError("simulated storage deletion failure")

    with monkeypatch.context() as scoped:
        scoped.setattr(LocalArtifactStorage, "delete", fail_delete)
        with pytest.raises(RegistryConsistencyError) as captured:
            registry.delete_version("classifier", "v1")
    assert registry.list_versions("classifier") == []
    storage = LocalArtifactStorage(tmp_path / "artifacts")
    assert captured.value.artifact_key == item.artifact_location
    assert storage.retrieve(item.artifact_location) == b"orphan"
    storage.delete(item.artifact_location)
    assert not storage.exists(item.artifact_location)


def test_json_schema_normalization_matches_persisted_metadata(registry):
    registry.register_model("classifier")
    item = registry.register_version(
        "classifier", "v1", ModelFramework.ONNX, b"data",
        input_schema={"choices": ("a", "b")},
    )
    assert item.input_schema == {"choices": ["a", "b"]}
    assert registry.get_version("classifier", "v1") == item
