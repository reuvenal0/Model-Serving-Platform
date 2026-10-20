from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from sqlalchemy.orm import sessionmaker

from app.core.exceptions import (
    ArtifactNotFoundError, DeploymentConflictError, DeploymentHealthCheckError,
    DeploymentPersistenceError, DeploymentRecoveryError, InvalidDeploymentStateError,
    ModelLoadingError, ModelVersionInUseError, ModelVersionNotFoundError,
)
from app.domain.deployment import Deployment
from app.domain.enums import DeploymentStatus, ModelFramework
from app.domain.resource_config import ResourceConfig
from app.infrastructure.database.database import create_sqlite_engine
from app.infrastructure.database.sqlite_deployment_repository import SQLiteDeploymentRepository
from app.infrastructure.execution.onnx_model_loader import OnnxModelLoader
from app.infrastructure.execution.sklearn_model_loader import SklearnModelLoader
from app.infrastructure.storage.local_artifact_storage import LocalArtifactStorage
from app.services.deployment_health_checker import DeploymentHealthChecker, ReadinessCheck
from app.services.deployment_manager import DeploymentManager
from app.services.model_loader_registry import ModelLoaderRegistry
from app.services.model_registry import ModelRegistry
from app.services.model_runtime_manager import ModelRuntimeManager


@pytest.fixture
def system(repository, engine, tmp_path, sklearn_artifact):
    storage = LocalArtifactStorage(tmp_path / "artifacts")
    registry = ModelRegistry(repository, storage)
    registry.register_model("model")
    registry.register_version("model", "v1", ModelFramework.SKLEARN, sklearn_artifact)
    deployments = SQLiteDeploymentRepository(sessionmaker(engine))
    sklearn = SklearnModelLoader(trusted=True)
    loaders = ModelLoaderRegistry({ModelFramework.SKLEARN: sklearn, ModelFramework.ONNX: OnnxModelLoader()})
    runtimes = ModelRuntimeManager(repository, storage, loaders, max_cache_size=1)
    checker = DeploymentHealthChecker()
    manager = DeploymentManager(deployments, repository, runtimes, checker)
    return SimpleNamespace(models=repository, storage=storage, registry=registry,
                           deployments=deployments, loaders=loaders, sklearn=sklearn,
                           runtimes=runtimes, checker=checker, manager=manager)


def deploy(system, identifier="d-1", **overrides):
    values = dict(model_name="model", version="v1", resources=ResourceConfig(1, 128),
                  deployment_id=identifier)
    return system.manager.deploy(**(values | overrides))


def test_real_sklearn_lifecycle_readiness_scale_and_history(system):
    item = deploy(system)
    assert item.status is DeploymentStatus.RUNNING
    health = system.manager.check_health(item.id)
    assert health.ready and health.kind == "load"
    smoke = system.manager.check_health(item.id, smoke_inputs=[[3], [4]])
    assert smoke.ready and smoke.kind == "smoke"
    np.testing.assert_allclose(system.runtimes.get_runtime("model", "v1").predict([[3], [4]]), [7, 9])
    scaled = system.manager.scale(item.id, 3)
    assert scaled.replicas == 3
    assert scaled.created_at == item.created_at
    assert scaled.updated_at >= item.updated_at
    persisted = system.deployments.get_deployment(item.id)
    assert persisted.deployment == scaled
    assert persisted.revision == 2
    stopped = system.manager.undeploy(item.id)
    assert stopped.status is DeploymentStatus.STOPPED
    assert not system.manager.check_health(item.id).ready
    assert system.manager.list_deployments("model") == [stopped]
    assert system.manager.list_deployments("missing") == []
    with pytest.raises(InvalidDeploymentStateError):
        system.manager.undeploy(item.id)
    with pytest.raises(InvalidDeploymentStateError):
        system.manager.scale(item.id, 4)


def test_real_onnx_load_readiness_and_model_specific_smoke(system, onnx_artifact):
    system.registry.register_version("model", "onnx-v1", ModelFramework.ONNX, onnx_artifact)
    item = deploy(system, version="onnx-v1", smoke_inputs={"X": [[1, 2]]})
    assert item.status is DeploymentStatus.RUNNING
    assert system.manager.check_health(item.id).ready
    np.testing.assert_allclose(system.runtimes.get_runtime("model", "onnx-v1").predict([[1, 2]])["Y"], [[3, 5]])


def test_two_deployments_share_runtime_and_stopping_one_preserves_other(system):
    first = deploy(system)
    runtime = system.runtimes.get_runtime("model", "v1")
    second = deploy(system, "d-2")
    assert first.id != second.id
    assert system.runtimes.get_runtime("model", "v1") is runtime
    assert system.runtimes.cache_info().loads == 1
    system.manager.undeploy(first.id)
    assert system.manager.check_health(second.id).ready
    assert system.runtimes.get_runtime("model", "v1") is runtime


def test_lru_eviction_is_not_deployment_state_and_health_reloads(system, onnx_artifact):
    first = deploy(system)
    old = system.runtimes.get_runtime("model", "v1")
    system.registry.register_version("model", "onnx-v1", ModelFramework.ONNX, onnx_artifact)
    deploy(system, "onnx", version="onnx-v1")
    assert system.runtimes.cache_info().evictions == 1
    assert system.manager.get_deployment(first.id).status is DeploymentStatus.RUNNING
    assert system.manager.check_health(first.id).ready
    assert system.runtimes.get_runtime("model", "v1") is not old
    assert system.runtimes.cache_info().loads == 3


def test_loading_error_is_persisted_as_failed(system):
    system.registry.register_version("model", "corrupt", ModelFramework.SKLEARN, b"invalid")
    with pytest.raises(ModelLoadingError):
        deploy(system, version="corrupt")
    assert system.manager.get_deployment("d-1").status is DeploymentStatus.FAILED
    assert not system.manager.check_health("d-1").ready


def test_smoke_failure_prevents_running_and_preserves_cause(system):
    with pytest.raises(DeploymentHealthCheckError) as captured:
        deploy(system, smoke_inputs=[[1, 2, 3]])
    assert captured.value.__cause__ is not None
    assert system.manager.get_deployment("d-1").status is DeploymentStatus.FAILED


def test_running_persistence_failure_falls_back_to_failed(system, monkeypatch):
    original = system.deployments.update_deployment
    failure = DeploymentPersistenceError("simulated RUNNING write failure")

    def update(item, *, expected_revision):
        if item.status is DeploymentStatus.RUNNING:
            raise failure
        return original(item, expected_revision=expected_revision)

    monkeypatch.setattr(system.deployments, "update_deployment", update)
    with pytest.raises(DeploymentPersistenceError) as captured:
        deploy(system)
    assert captured.value is failure
    assert system.manager.get_deployment("d-1").status is DeploymentStatus.FAILED


def test_failed_failure_persistence_is_explicit_and_reconcilable(system, monkeypatch):
    system.registry.register_version("model", "corrupt", ModelFramework.SKLEARN, b"bad")
    with monkeypatch.context() as scoped:
        scoped.setattr(system.deployments, "update_deployment", Mock(side_effect=DeploymentPersistenceError("disk full")))
        with pytest.raises(DeploymentRecoveryError) as captured:
            deploy(system, version="corrupt")
    assert isinstance(captured.value.operation_error, ModelLoadingError)
    assert isinstance(captured.value.persistence_error, DeploymentPersistenceError)
    assert system.manager.get_deployment("d-1").status is DeploymentStatus.DEPLOYING
    report = system.manager.reconcile()
    assert not report.failures
    assert report.health[0].kind == "interrupted"
    assert system.manager.get_deployment("d-1").status is DeploymentStatus.FAILED


def test_running_runtime_unavailable_marks_failed(system):
    item = deploy(system)
    key = system.registry.get_version("model", "v1").artifact_location
    system.storage.delete(key)  # Simulate out-of-band artifact loss.
    system.runtimes.invalidate("model", "v1")
    with pytest.raises(ArtifactNotFoundError):
        system.manager.check_health(item.id)
    assert system.manager.get_deployment(item.id).status is DeploymentStatus.FAILED


@pytest.mark.parametrize("modify", ["undeploy", "scale"])
def test_slow_deploy_cannot_activate_after_concurrent_state_change(system, monkeypatch, modify):
    entered, release = Event(), Event()
    original = system.sklearn.load

    def delayed(artifact):
        entered.set()
        assert release.wait(5)
        return original(artifact)

    monkeypatch.setattr(system.sklearn, "load", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(deploy, system)
        assert entered.wait(5)
        try:
            assert system.manager.get_deployment("d-1").status is DeploymentStatus.DEPLOYING
            changed = system.manager.undeploy("d-1") if modify == "undeploy" else system.manager.scale("d-1", 3)
        finally:
            release.set()
        with pytest.raises(DeploymentConflictError):
            pending.result(timeout=10)
    assert system.manager.get_deployment("d-1") == changed
    assert changed.status is not DeploymentStatus.RUNNING


def test_concurrent_distinct_deployments_share_loading(system, monkeypatch):
    entered, release, joined = Event(), Event(), Event()
    original = system.sklearn.load
    loads = Mock(wraps=original)

    def delayed(artifact):
        entered.set()
        assert release.wait(5)
        return loads(artifact)

    class ObservedFuture(Future):
        def result(self, timeout=None):
            joined.set()
            return super().result(timeout)

    monkeypatch.setattr(system.sklearn, "load", delayed)
    monkeypatch.setattr("app.services.model_runtime_manager.Future", ObservedFuture)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(deploy, system, "one")
        assert entered.wait(5)
        second = pool.submit(deploy, system, "two")
        try:
            assert joined.wait(5)
            assert len(system.manager.list_deployments()) == 2
        finally:
            release.set()
        assert first.result(timeout=10).status is DeploymentStatus.RUNNING
        assert second.result(timeout=10).status is DeploymentStatus.RUNNING
    loads.assert_called_once()
    assert system.runtimes.cache_info().loads == 1


def test_concurrent_scale_and_stop_have_one_revision_winner(system, database_url, monkeypatch):
    deploy(system)
    other_engine = create_sqlite_engine(database_url)
    other_repo = SQLiteDeploymentRepository(sessionmaker(other_engine))
    other_manager = DeploymentManager(other_repo, system.models, system.runtimes, system.checker)
    together = Barrier(2)

    def guarded(original):
        def get(identifier):
            snapshot = original(identifier)
            together.wait(timeout=5)
            return snapshot
        return get

    monkeypatch.setattr(system.deployments, "get_deployment", guarded(system.deployments.get_deployment))
    monkeypatch.setattr(other_repo, "get_deployment", guarded(other_repo.get_deployment))

    def operate(operation):
        try:
            return operation()
        except DeploymentConflictError as error:
            return error

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            scale = pool.submit(operate, lambda: system.manager.scale("d-1", 3))
            stop = pool.submit(operate, lambda: other_manager.undeploy("d-1"))
            results = [scale.result(timeout=10), stop.result(timeout=10)]
        winners = [value for value in results if isinstance(value, Deployment)]
        assert len(winners) == 1
        assert len([value for value in results if isinstance(value, DeploymentConflictError)]) == 1
        reader = SQLiteDeploymentRepository(sessionmaker(other_engine))
        assert reader.get_deployment("d-1").deployment == winners[0]
        assert reader.get_deployment("d-1").revision == 2
    finally:
        other_engine.dispose()


def test_health_check_cannot_report_ready_after_stop(system, monkeypatch):
    deploy(system)
    entered, release = Event(), Event()

    def delayed(runtime, *, smoke_inputs):
        entered.set()
        assert release.wait(5)
        return ReadinessCheck("load")

    monkeypatch.setattr(system.checker, "check", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(system.manager.check_health, "d-1")
        assert entered.wait(5)
        try:
            system.manager.undeploy("d-1")
        finally:
            release.set()
        with pytest.raises(DeploymentConflictError):
            pending.result(timeout=10)
    assert not system.manager.check_health("d-1").ready


def test_restart_reconciliation_reloads_running_and_marks_interrupted(system, database_url):
    deploy(system)
    system.deployments.save_deployment(Deployment("interrupted", "model", "v1", 1, ResourceConfig(1, 128)))
    other_engine = create_sqlite_engine(database_url)
    try:
        repo = SQLiteDeploymentRepository(sessionmaker(other_engine))
        runtimes = ModelRuntimeManager(system.models, system.storage, system.loaders)
        restarted = DeploymentManager(repo, system.models, runtimes, DeploymentHealthChecker())
        assert restarted.get_deployment("d-1").status is DeploymentStatus.RUNNING
        assert runtimes.cache_info().loads == 0  # Persisted RUNNING is not process readiness.
        report = restarted.reconcile()
        assert not report.failures
        health = {item.deployment_id: item for item in report.health}
        assert health["d-1"].ready
        assert not health["interrupted"].ready
        assert health["interrupted"].status is DeploymentStatus.FAILED
        assert runtimes.cache_info().loads == 1
    finally:
        other_engine.dispose()


def test_reconciliation_collects_errors_and_continues(system, onnx_artifact):
    deploy(system)
    system.registry.register_version("model", "onnx", ModelFramework.ONNX, onnx_artifact)
    deploy(system, "good", version="onnx")
    system.storage.delete(system.registry.get_version("model", "v1").artifact_location)
    system.runtimes.clear()
    report = system.manager.reconcile()
    assert len(report.failures) == 1
    assert report.failures[0].deployment_id == "d-1"
    assert isinstance(report.failures[0].error, ArtifactNotFoundError)
    assert report.health[0].deployment_id == "good" and report.health[0].ready
    assert system.manager.get_deployment("d-1").status is DeploymentStatus.FAILED


@pytest.mark.parametrize("stop", [False, True])
def test_registry_deletion_preserves_artifact_and_history(system, stop):
    deploy(system)
    if stop:
        system.manager.undeploy("d-1")
    before = system.registry.download_artifact("model", "v1")
    with pytest.raises(ModelVersionInUseError):
        system.registry.delete_version("model", "v1")
    assert system.registry.download_artifact("model", "v1") == before
    assert system.manager.get_deployment("d-1").model_name == "model"


def test_deleting_version_during_slow_deploy_is_blocked(system, monkeypatch):
    entered, release = Event(), Event()
    original = system.sklearn.load

    def delayed(artifact):
        entered.set()
        assert release.wait(5)
        return original(artifact)

    monkeypatch.setattr(system.sklearn, "load", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(deploy, system)
        assert entered.wait(5)
        try:
            with pytest.raises(ModelVersionInUseError):
                system.registry.delete_version("model", "v1")
        finally:
            release.set()
        assert pending.result(timeout=10).status is DeploymentStatus.RUNNING


def test_delete_winning_after_eligibility_check_prevents_deployment_insert(system, monkeypatch):
    entered, release = Event(), Event()
    original = system.deployments.save_deployment

    def delayed(item):
        entered.set()
        assert release.wait(5)
        return original(item)

    monkeypatch.setattr(system.deployments, "save_deployment", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(deploy, system)
        assert entered.wait(5)
        try:
            system.registry.delete_version("model", "v1")
        finally:
            release.set()
        with pytest.raises(ModelVersionNotFoundError):
            pending.result(timeout=10)
    assert system.manager.list_deployments() == []
    assert system.runtimes.cache_info().loads == 0


def test_failed_health_persistence_does_not_claim_health_and_reports_unverified_state(system, monkeypatch):
    deploy(system)
    key = system.registry.get_version("model", "v1").artifact_location
    system.storage.delete(key)
    system.runtimes.clear()
    with monkeypatch.context() as scoped:
        scoped.setattr(system.deployments, "update_deployment", Mock(side_effect=DeploymentPersistenceError("disk full")))
        with pytest.raises(DeploymentRecoveryError) as captured:
            system.manager.check_health("d-1")
    assert isinstance(captured.value.operation_error, ArtifactNotFoundError)
    assert system.manager.get_deployment("d-1").status is DeploymentStatus.RUNNING
    report = system.manager.reconcile()
    assert len(report.failures) == 1
    assert system.manager.get_deployment("d-1").status is DeploymentStatus.FAILED


def test_stale_failed_health_cannot_overwrite_stopped_state(system, monkeypatch):
    deploy(system)
    entered, release = Event(), Event()
    failure = DeploymentHealthCheckError("smoke failed during concurrent stop")

    def delayed(runtime, *, smoke_inputs):
        entered.set()
        assert release.wait(5)
        raise failure

    monkeypatch.setattr(system.checker, "check", delayed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(system.manager.check_health, "d-1")
        assert entered.wait(5)
        try:
            system.manager.undeploy("d-1")
        finally:
            release.set()
        with pytest.raises(DeploymentRecoveryError) as captured:
            pending.result(timeout=10)
    assert captured.value.operation_error is failure
    assert isinstance(captured.value.persistence_error, DeploymentConflictError)
    assert system.manager.get_deployment("d-1").status is DeploymentStatus.STOPPED


def test_reconciliation_reports_failed_persistence_and_keeps_inactive_history(system, monkeypatch):
    item = deploy(system)
    system.manager.undeploy(item.id)
    system.deployments.save_deployment(Deployment("pending", "model", "v1", 1, ResourceConfig(1, 128)))
    with monkeypatch.context() as scoped:
        scoped.setattr(system.deployments, "update_deployment", Mock(side_effect=DeploymentPersistenceError("disk full")))
        report = system.manager.reconcile()
    assert len(report.failures) == 1
    assert report.failures[0].deployment_id == "pending"
    assert isinstance(report.failures[0].error, DeploymentPersistenceError)
    assert report.health[0].status is DeploymentStatus.STOPPED
    assert not report.health[0].ready
    assert system.manager.get_deployment("pending").status is DeploymentStatus.DEPLOYING
