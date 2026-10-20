from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest
from sqlalchemy import delete, insert, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.core.exceptions import (
    DeploymentConflictError, DeploymentNotFoundError, DeploymentPersistenceError,
    DomainValidationError, DuplicateDeploymentError, ModelVersionNotFoundError,
)
from app.domain.deployment import Deployment
from app.domain.enums import DeploymentStatus, ModelFramework
from app.domain.model import Model
from app.domain.model_version import ModelVersion
from app.domain.resource_config import ResourceConfig
from app.infrastructure.database.database import create_sqlite_engine, initialize_database
from app.infrastructure.database.records import DeploymentRecord, ModelVersionRecord
from app.infrastructure.database.sqlite_deployment_repository import SQLiteDeploymentRepository


@pytest.fixture
def deployments(engine, repository):
    for name in ["model", "other"]:
        repository.save_model(Model(name))
        repository.save_version(ModelVersion(name, "v1", f"{name}/key", ModelFramework.ONNX))
    return SQLiteDeploymentRepository(sessionmaker(engine))


def entity(**overrides):
    values = dict(id="d-1", model_name="model", version="v1", replicas=1,
                  resources=ResourceConfig(0.5, 128), endpoint=None)
    return Deployment(**(values | overrides))


def test_save_retrieve_update_and_filter_history(deployments):
    original = entity()
    saved = deployments.save_deployment(original)
    assert saved.revision == 0
    assert saved.deployment == original
    loaded = deployments.get_deployment(original.id)
    assert loaded.deployment == original
    assert loaded.deployment is not original
    loaded.deployment.activate()
    updated = deployments.update_deployment(loaded.deployment, expected_revision=loaded.revision)
    assert updated.revision == 1
    assert updated.deployment.status is DeploymentStatus.RUNNING
    assert deployments.get_deployment("d-1") == updated
    deployments.save_deployment(entity(id="d-2", model_name="other"))
    assert [item.deployment.id for item in deployments.list_deployments()] == ["d-1", "d-2"]
    assert [item.deployment.id for item in deployments.list_deployments("model")] == ["d-1"]
    assert deployments.list_deployments("missing") == []


def test_missing_and_duplicate_ids(deployments):
    with pytest.raises(DeploymentNotFoundError):
        deployments.get_deployment("missing")
    with pytest.raises(DeploymentNotFoundError):
        deployments.update_deployment(entity(), expected_revision=0)
    original = deployments.save_deployment(entity())
    with pytest.raises(DuplicateDeploymentError):
        deployments.save_deployment(entity(replicas=2))
    assert deployments.get_deployment("d-1") == original


def test_foreign_key_requires_registered_version(deployments):
    with pytest.raises(ModelVersionNotFoundError):
        deployments.save_deployment(entity(version="missing"))
    assert deployments.list_deployments() == []


def test_database_foreign_key_rejects_orphans_even_without_repository(engine):
    now = datetime.now(timezone.utc)
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(insert(DeploymentRecord).values(
                id="orphan", model_name="missing", version="v1", replicas=1,
                cpu=1, memory_mb=128, status="deploying", endpoint=None,
                created_at=now, updated_at=now, revision=0,
            ))


@pytest.mark.parametrize("status", list(DeploymentStatus))
def test_foreign_key_protects_all_deployment_history(deployments, engine, status):
    deployments.save_deployment(entity(status=status))
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(delete(ModelVersionRecord).where(
                ModelVersionRecord.model_name == "model", ModelVersionRecord.version == "v1",
            ))
    assert deployments.get_deployment("d-1").deployment.status is status


def test_stale_write_does_not_restore_running_after_stop(deployments):
    initial = deployments.save_deployment(entity())
    stale = deployments.get_deployment("d-1")
    initial.deployment.stop()
    stopped = deployments.update_deployment(initial.deployment, expected_revision=initial.revision)
    stale.deployment.activate()
    with pytest.raises(DeploymentConflictError):
        deployments.update_deployment(stale.deployment, expected_revision=stale.revision)
    assert deployments.get_deployment("d-1") == stopped


@pytest.mark.parametrize("field,value", [
    ("model_name", "other"), ("version", "different"),
    ("created_at", datetime(2000, 1, 1, tzinfo=timezone.utc)),
])
def test_identity_and_creation_timestamp_cannot_be_reassigned(deployments, field, value):
    original = deployments.save_deployment(entity())
    changed = replace(original.deployment, **{field: value})
    with pytest.raises(DeploymentConflictError):
        deployments.update_deployment(changed, expected_revision=0)
    assert deployments.get_deployment("d-1") == original


@pytest.mark.parametrize("revision", [-1, True, 0.5, None])
def test_invalid_revision_is_validation_failure(deployments, revision):
    with pytest.raises(DomainValidationError):
        deployments.update_deployment(entity(), expected_revision=revision)


def test_persistence_across_fresh_engine_and_timezone_round_trip(deployments, database_url):
    local = datetime(2026, 1, 1, 12, 4, 5, 123456,
                     tzinfo=timezone(timedelta(hours=2)))
    item = entity(created_at=local, updated_at=local, endpoint="logical endpoint metadata")
    saved = deployments.save_deployment(item)
    engine = create_sqlite_engine(database_url)
    try:
        initialize_database(engine)
        fresh = SQLiteDeploymentRepository(sessionmaker(engine))
        restored = fresh.get_deployment("d-1")
        assert restored == saved
        assert restored.deployment.resources == ResourceConfig(0.5, 128)
        assert restored.deployment.created_at.tzinfo is timezone.utc
        assert restored.deployment.created_at.hour == 10
        assert restored.deployment.updated_at.microsecond == 123456
        restored.deployment.scale(3)
        assert fresh.update_deployment(restored.deployment, expected_revision=0).revision == 1
        assert deployments.get_deployment("d-1").deployment.replicas == 3
    finally:
        engine.dispose()


def test_concurrent_compare_and_swap_across_independent_engines(deployments, database_url):
    deployments.save_deployment(entity())
    engines = [create_sqlite_engine(database_url), create_sqlite_engine(database_url)]
    together = Barrier(2)

    def scale(engine, replicas):
        repo = SQLiteDeploymentRepository(sessionmaker(engine))
        snapshot = repo.get_deployment("d-1")
        snapshot.deployment.scale(replicas)
        together.wait(timeout=5)
        try:
            return repo.update_deployment(snapshot.deployment, expected_revision=snapshot.revision)
        except DeploymentConflictError as error:
            return error

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(scale, engines[0], 2)
            second = pool.submit(scale, engines[1], 3)
            results = [first.result(timeout=10), second.result(timeout=10)]
        winners = [value for value in results if not isinstance(value, Exception)]
        assert len(winners) == 1
        assert len([value for value in results if isinstance(value, DeploymentConflictError)]) == 1
        assert deployments.get_deployment("d-1") == winners[0]
        assert winners[0].revision == 1
    finally:
        for engine in engines:
            engine.dispose()


def test_backend_errors_and_corrupt_records_are_translated(deployments, engine):
    empty_engine = create_sqlite_engine("sqlite:///:memory:")
    try:
        with pytest.raises(DeploymentPersistenceError):
            SQLiteDeploymentRepository(sessionmaker(empty_engine)).list_deployments()
    finally:
        empty_engine.dispose()
    deployments.save_deployment(entity())
    with engine.begin() as connection:
        connection.execute(text("UPDATE deployments SET created_at='2026-01-01T12:00:00'"))
    with pytest.raises(DeploymentPersistenceError):
        deployments.get_deployment("d-1")


def test_initialization_adds_missing_deployment_table_without_rewriting_model_data(deployments, repository, engine):
    # Simulate a Phase 3 database: model tables/data exist, deployments does not.
    DeploymentRecord.__table__.drop(engine)
    before = repository.get_version("model", "v1")
    initialize_database(engine)
    assert repository.get_version("model", "v1") == before
    assert deployments.list_deployments() == []
    assert deployments.save_deployment(entity()).revision == 0
