"""Model registration coordinating independent metadata and artifact backends."""

import json
from typing import Any
from uuid import uuid4

from app.core.exceptions import (
    ArtifactCleanupError, DomainValidationError, DuplicateModelError,
    DuplicateVersionError, ModelNotFoundError, ModelVersionNotFoundError,
    RegistryConsistencyError,
)
from app.domain.enums import ModelFramework
from app.domain.model import Model
from app.domain.model_version import ModelVersion
from app.interfaces.artifact_storage import ArtifactStorage
from app.interfaces.model_repository import ModelRepository


class ModelRegistry:
    def __init__(self, repository: ModelRepository, storage: ArtifactStorage) -> None:
        self._repository = repository
        self._storage = storage

    def register_model(self, name: str, description: str = "") -> Model:
        model = Model(name, description)
        try:
            self._repository.get_model(name)
        except ModelNotFoundError:
            pass
        else:
            raise DuplicateModelError(f"Model {name!r} already exists")
        # The repository constraint also protects against concurrent registration.
        self._repository.save_model(model)
        return model

    @staticmethod
    def _schema(value: dict[str, Any] | None) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise DomainValidationError("Schemas must be dictionaries of JSON metadata")
        try:
            # Normalize JSON-compatible containers before returning/persisting metadata.
            return json.loads(json.dumps(value, allow_nan=False))
        except (TypeError, ValueError, OverflowError) as error:
            raise DomainValidationError("Schemas must contain JSON-serializable metadata") from error

    def register_version(
        self, model_name: str, version: str, framework: ModelFramework,
        content: bytes, *, input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> ModelVersion:
        """Validate, check identities, create bytes, then commit metadata.

        Only a successfully created artifact is eligible for compensation. A
        repository failure is re-raised after cleanup; double failures expose
        both errors and the artifact key via RegistryConsistencyError.
        """
        key = f"artifacts/{uuid4().hex}"
        metadata = ModelVersion(
            model_name=model_name, version=version, artifact_location=key,
            framework=framework, input_schema=self._schema(input_schema),
            output_schema=self._schema(output_schema),
        )
        if not isinstance(content, bytes):
            raise DomainValidationError("Artifact content must be bytes")
        self._repository.get_model(model_name)
        try:
            self._repository.get_version(model_name, version)
        except ModelVersionNotFoundError:
            pass
        else:
            raise DuplicateVersionError(f"Version {model_name!r}/{version!r} already exists")

        try:
            self._storage.save(key, content)
        except ArtifactCleanupError as error:
            # Publication may have succeeded; do not pretend an ordinary rollback occurred.
            raise RegistryConsistencyError(
                "Artifact creation cleanup failed; metadata was not saved. "
                "Inspect the published flag and temporary path in the original error.",
                key, operation_error=error,
            ) from error
        try:
            self._repository.save_version(metadata)
        except Exception as operation_error:
            try:
                self._storage.delete(key)
            except Exception as cleanup_error:
                raise RegistryConsistencyError(
                    "Version registration failed and artifact rollback failed; "
                    "inspect storage and remove any orphaned artifact.",
                    key, operation_error, cleanup_error,
                ) from cleanup_error
            raise
        return metadata

    def get_model(self, name: str) -> Model:
        return self._repository.get_model(name)

    def list_models(self) -> list[Model]:
        return self._repository.list_models()

    def list_versions(self, model_name: str) -> list[ModelVersion]:
        return self._repository.list_versions(model_name)

    def get_version(self, model_name: str, version: str) -> ModelVersion:
        return self._repository.get_version(model_name, version)

    def download_artifact(self, model_name: str, version: str) -> bytes:
        metadata = self._repository.get_version(model_name, version)
        return self._storage.retrieve(metadata.artifact_location)

    def delete_version(self, model_name: str, version: str) -> None:
        """Commit metadata deletion before removing bytes; never delete the parent.

        A metadata failure leaves bytes alone. A subsequent artifact failure
        reports an orphan key for explicit storage cleanup, including a missing
        artifact that indicates prior inconsistency. No deployment checks exist.
        """
        self._repository.get_version(model_name, version)
        # Use the snapshot actually removed by the repository, not an earlier read.
        metadata = self._repository.delete_version(model_name, version)
        try:
            self._storage.delete(metadata.artifact_location)
        except Exception as error:
            raise RegistryConsistencyError(
                "Version metadata was deleted but artifact deletion failed; "
                "inspect storage and retry artifact cleanup directly.",
                metadata.artifact_location, cleanup_error=error,
            ) from error
