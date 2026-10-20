"""Controlled offline smoke data; refuses the normal development database."""

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import pickle
import sqlite3
import sys
from sqlalchemy.engine import make_url

from app.config import Settings
from app.core.container import ServiceContainer
from app.domain.deployment import Deployment
from app.domain.enums import ModelFramework
from app.domain.resource_config import ResourceConfig


def require_isolated_storage() -> Settings:
    settings = Settings()
    if (
        os.getenv("MSP_CONTAINER_SMOKE") != "1"
        or settings.database_url != "sqlite:////data/db/smoke.db"
        or settings.artifact_storage_dir != Path("/data/artifacts")
    ):
        raise ValueError(
            "This fixture only accepts explicitly isolated container smoke storage"
        )
    return settings


def fingerprint() -> dict:
    settings = require_isolated_storage()
    database = Path(make_url(settings.database_url).database).resolve()
    with closing(
        sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    ) as connection:
        versions = connection.execute(
            "SELECT model_name, version, created_at, artifact_location FROM model_versions ORDER BY model_name, version"
        ).fetchall()
        history = connection.execute(
            "SELECT * FROM routing_history ORDER BY model_name, revision"
        ).fetchall()
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    artifacts = sorted(settings.artifact_storage_dir.glob("*.artifact"))
    if len(artifacts) != 1 or len(versions) != 1:
        raise ValueError("Unexpected smoke storage contents")
    return {
        "artifact_sha256": hashlib.sha256(artifacts[0].read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(json.dumps(versions).encode()).hexdigest(),
        "history_sha256": hashlib.sha256(json.dumps(history).encode()).hexdigest(),
        "tables": [row[0] for row in tables],
    }


def seed() -> None:
    settings = require_isolated_storage()
    if not settings.trust_sklearn_artifacts:
        raise ValueError(
            "Trusted locally generated smoke artifacts require explicit opt-in"
        )
    from sklearn.linear_model import LinearRegression

    artifact = pickle.dumps(
        LinearRegression().fit([[1.0], [2.0], [3.0]], [3.0, 5.0, 7.0])
    )
    with ServiceContainer(settings) as container:
        if container.model_registry.list_models():
            raise ValueError(
                "Smoke seeding requires empty model metadata; nothing was overwritten"
            )
        registry = container.model_registry
        registry.register_model("container-smoke")
        registry.register_version(
            "container-smoke", "v1", ModelFramework.SKLEARN, artifact
        )
        container.deployment_manager.deploy(
            "container-smoke",
            "v1",
            ResourceConfig(1, 128),
            deployment_id="smoke-running",
        )
        container.deployment_repository.save_deployment(
            Deployment(
                "smoke-interrupted", "container-smoke", "v1", 1, ResourceConfig(1, 128)
            )
        )
        container.routing_manager.configure_routing("container-smoke", {"v1": 100}, 0)
        result = container.inference_service.predict("container-smoke", "v1", [[4.0]])
        if abs(result.predictions[0] - 9.0) > 1e-8:
            raise ValueError("Unexpected controlled model output")
    print(json.dumps(fingerprint()))


if __name__ == "__main__":
    if sys.argv[1:] == ["seed"]:
        seed()
    elif sys.argv[1:] == ["fingerprint"]:
        print(json.dumps(fingerprint()))
    else:
        raise SystemExit("Choose seed or fingerprint")
