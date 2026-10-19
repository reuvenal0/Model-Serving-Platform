from hashlib import sha256
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.exceptions import (
    ArtifactCleanupError, ArtifactNotFoundError, ArtifactStorageError,
    DomainValidationError, DuplicateArtifactError,
)
from app.infrastructure.storage.local_artifact_storage import LocalArtifactStorage


@pytest.fixture
def storage(tmp_path):
    return LocalArtifactStorage(tmp_path / "artifacts")


@pytest.mark.parametrize("content", [b"", b"harmless\x00\xff\x01", bytes(range(256)) * 4096], ids=["empty", "binary", "one-megabyte"])
def test_content_integrity_and_persistence(storage, tmp_path, content):
    assert not storage.exists("models/version/artifact.bin")
    assert storage.save("models/version/artifact.bin", content) is None
    assert storage.exists("models/version/artifact.bin")
    assert storage.retrieve("models/version/artifact.bin") == content
    another = LocalArtifactStorage(tmp_path / "artifacts")
    assert another.retrieve("models/version/artifact.bin") == content
    assert len(list((tmp_path / "artifacts").iterdir())) == 1


def test_delete_artifact(storage):
    storage.save("model/v1", b"artifact")
    assert storage.delete("model/v1") is None
    assert not storage.exists("model/v1")
    with pytest.raises(ArtifactNotFoundError):
        storage.retrieve("model/v1")


def test_duplicate_key_preserves_original_and_cleans_temporary_file(storage, tmp_path):
    storage.save("model/v1", b"original")
    with pytest.raises(DuplicateArtifactError):
        storage.save("model/v1", b"replacement")
    assert storage.retrieve("model/v1") == b"original"
    assert not list((tmp_path / "artifacts").glob(".upload-*"))


@pytest.mark.parametrize("key", [
    "", " ", ".", "..", "../outside", "a/../../outside", "/absolute", "a//b",
    "a/./b", "a/../b", "a/", "C:/absolute", "C:relative", "a\\..\\outside",
    "\\server\\share", "a:stream", "a\x00b", "%2e%2e/outside", "a.", "x" * 1025,
])
@pytest.mark.parametrize("operation", ["save", "retrieve", "delete", "exists"])
def test_unsafe_keys_rejected(storage, key, operation):
    with pytest.raises(DomainValidationError):
        if operation == "save":
            storage.save(key, b"data")
        else:
            getattr(storage, operation)(key)


@pytest.mark.parametrize("operation", ["retrieve", "delete"])
def test_missing_artifact(storage, operation):
    with pytest.raises(ArtifactNotFoundError):
        getattr(storage, operation)("missing")


def test_invalid_content(storage):
    with pytest.raises(DomainValidationError):
        storage.save("model/v1", "not bytes")
    assert not storage.exists("model/v1")


def test_write_failure_leaves_no_partial_artifact(storage, tmp_path):
    with patch("app.infrastructure.storage.local_artifact_storage.os.fsync", side_effect=OSError("disk full")):
        with pytest.raises(ArtifactStorageError):
            storage.save("model/v1", b"data")
    assert not storage.exists("model/v1")
    assert list((tmp_path / "artifacts").iterdir()) == []


def test_publication_failure_leaves_no_artifact(storage, tmp_path):
    with patch.object(LocalArtifactStorage, "_publish", side_effect=OSError("publication failed")):
        with pytest.raises(ArtifactStorageError):
            storage.save("model/v1", b"data")
    assert not storage.exists("model/v1")
    assert list((tmp_path / "artifacts").iterdir()) == []


def test_cleanup_failure_is_explicit(storage):
    with patch.object(LocalArtifactStorage, "_publish", side_effect=OSError("publication failed")):
        with patch.object(Path, "unlink", side_effect=PermissionError("cannot clean temp file")):
            with pytest.raises(ArtifactCleanupError) as captured:
                storage.save("model/v1", b"data")
    assert captured.value.published is False
    assert captured.value.artifact_key == "model/v1"
    assert Path(captured.value.temporary_file).exists()
    assert not storage.exists("model/v1")
    Path(captured.value.temporary_file).unlink()


def test_existing_directory_target_is_rejected(storage, tmp_path):
    target = tmp_path / "artifacts" / (sha256(b"model/v1").hexdigest() + ".artifact")
    target.mkdir()
    with pytest.raises(ArtifactStorageError):
        storage.save("model/v1", b"data")
    assert target.is_dir()


def test_symlink_target_cannot_access_external_file(storage, tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"untouched")
    target = tmp_path / "artifacts" / (sha256(b"model/v1").hexdigest() + ".artifact")
    try:
        os.symlink(outside, target)
    except OSError:
        pytest.skip("Creating symlinks requires privileges on this host")
    for operation in [storage.retrieve, storage.delete, storage.exists]:
        with pytest.raises(ArtifactStorageError):
            operation("model/v1")
    with pytest.raises(ArtifactStorageError):
        storage.save("model/v1", b"replacement")
    assert outside.read_bytes() == b"untouched"


def test_concurrent_saves_publish_one_complete_artifact(storage, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    payloads = [b"first" * 1000, b"second" * 1000]

    def attempt(content):
        try:
            storage.save("shared/key", content)
            return content
        except DuplicateArtifactError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, payloads))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert storage.retrieve("shared/key") == winners[0]
    assert not list((tmp_path / "artifacts").glob(".upload-*"))
