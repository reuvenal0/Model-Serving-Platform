"""Create-only local artifact storage with atomic publication."""

from hashlib import sha256
import os
from pathlib import Path
import re
import stat
import tempfile

from app.core.exceptions import (
    ArtifactCleanupError, ArtifactNotFoundError, ArtifactStorageError, ArtifactTooLargeError,
    DomainValidationError, DuplicateArtifactError,
)
from app.interfaces.artifact_storage import ArtifactStorage


class LocalArtifactStorage(ArtifactStorage):
    """Map safe logical keys to flat hashed files under a trusted private root.

    No caller-controlled path component is used as a physical filename. Complete
    temporary files are published with a non-replacing rename on Windows and
    a non-replacing hard link on POSIX, on the same filesystem.
    Administrators must prevent untrusted processes from modifying this root;
    portable path checks cannot defend against a hostile filesystem owner.
    """

    def __init__(self, root: str | Path) -> None:
        try:
            self._root = Path(root).resolve()
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise ArtifactStorageError("Could not initialize artifact storage") from error

    def _path(self, key: str) -> Path:
        if not isinstance(key, str) or not key or len(key) > 1024:
            raise DomainValidationError("Artifact key must contain 1 to 1024 characters")
        parts = key.split("/")
        if any(
            not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", part)
            or part.endswith(".")
            for part in parts
        ):
            raise DomainValidationError("Artifact key must use safe relative slash-separated names")
        path = self._root / (sha256(key.encode("utf-8")).hexdigest() + ".artifact")
        # Also reject reparse points (including Windows junctions) at the target.
        if self._root.resolve() != self._root:
            raise ArtifactStorageError("Storage root has been redirected")
        try:
            info = path.lstat()
        except FileNotFoundError:
            return path
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
        ):
            raise ArtifactStorageError("Artifact target must not be a symbolic link or reparse point")
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactStorageError("Artifact target must be a regular file")
        return path

    @staticmethod
    def _publish(temporary: Path, target: Path) -> None:
        if os.name == "nt":
            # Windows rename fails if the destination already exists.
            os.rename(temporary, target)
        else:
            # POSIX rename would replace existing content; link is create-only.
            os.link(temporary, target)

    def save(self, key: str, content: bytes) -> None:
        if not isinstance(content, bytes):
            raise DomainValidationError("Artifact content must be bytes")
        temporary: Path | None = None
        published = False
        try:
            target = self._path(key)
            descriptor, name = tempfile.mkstemp(prefix=".upload-", dir=self._root)
            temporary = Path(name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                self._publish(temporary, target)
            except FileExistsError as error:
                raise DuplicateArtifactError(f"Artifact {key!r} already exists") from error
            published = True
            if os.name == "nt":
                temporary = None  # Rename consumed the temporary file.
        except OSError as error:
            raise ArtifactStorageError(f"Could not save artifact {key!r}") from error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError as error:
                    # The exception context preserves any earlier write/publication failure.
                    raise ArtifactCleanupError(key, str(temporary), published) from error

    def retrieve(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError as error:
            raise ArtifactNotFoundError(f"Artifact {key!r} does not exist") from error
        except OSError as error:
            raise ArtifactStorageError(f"Could not retrieve artifact {key!r}") from error

    def delete(self, key: str) -> None:
        try:
            self._path(key).unlink()
        except FileNotFoundError as error:
            raise ArtifactNotFoundError(f"Artifact {key!r} does not exist") from error
        except OSError as error:
            raise ArtifactStorageError(f"Could not delete artifact {key!r}") from error

    def retrieve_limited(self, key: str, *, max_bytes: int) -> bytes:
        self._validate_limit(max_bytes)
        try:
            with self._path(key).open("rb") as stream:
                # Inspect the opened file, not a separate path lookup which may
                # refer to a different file after concurrent replacement.
                if os.fstat(stream.fileno()).st_size > max_bytes:
                    raise ArtifactTooLargeError("Artifact exceeds maximum runtime loading size")
                chunks = []
                total = 0
                while True:
                    chunk = stream.read(min(64 * 1024, max_bytes - total + 1))
                    if not chunk:
                        return b"".join(chunks)
                    total += len(chunk)
                    if total > max_bytes:
                        raise ArtifactTooLargeError("Artifact exceeds maximum runtime loading size")
                    chunks.append(chunk)
        except FileNotFoundError as error:
            raise ArtifactNotFoundError("Runtime artifact does not exist") from error
        except OSError as error:
            raise ArtifactStorageError("Could not read runtime artifact") from error

    def exists(self, key: str) -> bool:
        try:
            return self._path(key).is_file()
        except OSError as error:
            raise ArtifactStorageError(f"Could not check artifact {key!r}") from error
