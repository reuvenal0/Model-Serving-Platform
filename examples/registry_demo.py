"""Run after installing the project: python examples/registry_demo.py."""

from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.domain.enums import ModelFramework
from app.infrastructure.database.database import create_sqlite_engine, initialize_database
from app.infrastructure.database.sqlite_model_repository import SQLiteModelRepository
from app.infrastructure.storage.local_artifact_storage import LocalArtifactStorage
from app.services.model_registry import ModelRegistry


def main() -> None:
    with TemporaryDirectory(prefix="registry-demo-") as directory:
        root = Path(directory)
        settings = Settings(
            database_url=URL.create("sqlite", database=str(root / "metadata.db")).render_as_string(),
            artifact_storage_dir=root / "artifacts",
        )
        engine = create_sqlite_engine(settings.database_url)
        try:
            initialize_database(engine)
            repository = SQLiteModelRepository(sessionmaker(engine))
            storage = LocalArtifactStorage(settings.artifact_storage_dir)
            registry = ModelRegistry(repository, storage)
            registry.register_model("demo-model", "Harmless registry demonstration")
            content = b"Harmless binary artifact data\x00\x01"
            registered = registry.register_version(
                "demo-model", "v1", ModelFramework.SKLEARN, content,
                input_schema={"type": "array"}, output_schema={"type": "number"},
            )
            print("Models:", [model.name for model in registry.list_models()])
            print("Versions:", [version.version for version in registry.list_versions("demo-model")])
            print("Saved logical artifact key:", registered.artifact_location)
            retrieved = registry.download_artifact("demo-model", "v1")
            if retrieved != content:
                raise RuntimeError("Retrieved artifact bytes do not match")
            print(f"Artifact integrity verified: {len(retrieved)} bytes match exactly.")
            print("Demo succeeded; temporary metadata and artifacts will be removed.")
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
