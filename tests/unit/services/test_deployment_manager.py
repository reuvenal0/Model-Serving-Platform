from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.core.exceptions import (
    DeploymentConflictError, DeploymentHealthCheckError, DeploymentPersistenceError,
    DeploymentRecoveryError, DomainValidationError, DuplicateDeploymentError,
    InvalidDeploymentStateError, ModelLoadingError, ModelNotFoundError,
    ModelVersionNotFoundError,
)
from app.domain.deployment import Deployment
from app.domain.enums import DeploymentStatus, ModelFramework, ModelVersionStatus
from app.domain.model import Model
from app.domain.model_version import ModelVersion
from app.domain.resource_config import ResourceConfig
from app.interfaces.deployment_repository import DeploymentRepository, DeploymentSnapshot
from app.interfaces.model_repository import ModelRepository
from app.services.deployment_health_checker import DeploymentHealthChecker, ReadinessCheck
from app.services.deployment_manager import DeploymentManager
from app.services.model_runtime_manager import ModelRuntimeManager


@pytest.fixture
def dependencies():
    deployments = Mock(spec=DeploymentRepository)
    models = Mock(spec=ModelRepository)
    models.get_model.return_value = Model("model")
    models.get_version.return_value = ModelVersion("model", "v1", "key", ModelFramework.ONNX)
    runtimes = Mock(spec=ModelRuntimeManager)
    checker = Mock(spec=DeploymentHealthChecker)
    checker.check.return_value = ReadinessCheck("load")
    deployments.save_deployment.side_effect = lambda item: DeploymentSnapshot(replace(item), 0)
    deployments.update_deployment.side_effect = lambda item, expected_revision: DeploymentSnapshot(
        replace(item), expected_revision + 1,
    )
    manager = DeploymentManager(deployments, models, runtimes, checker)
    return SimpleNamespace(deployments=deployments, models=models, runtimes=runtimes,
                           checker=checker, manager=manager)


def deploy(d, **overrides):
    values = dict(model_name="model", version="v1", resources=ResourceConfig(1, 128),
                  deployment_id="deployment-1")
    return d.manager.deploy(**(values | overrides))


def snapshot(d, status=DeploymentStatus.RUNNING):
    value = DeploymentSnapshot(Deployment(
        "deployment-1", "model", "v1", 1, ResourceConfig(1, 128), status=status,
    ), 4)
    d.deployments.get_deployment.return_value = value
    return value


def test_successful_deploy_persists_initial_and_activated_snapshots(dependencies):
    d = dependencies
    result = deploy(d)
    initial = d.deployments.save_deployment.call_args.args[0]
    assert initial.status is DeploymentStatus.DEPLOYING
    assert result.status is DeploymentStatus.RUNNING
    assert initial is not result
    assert initial.created_at == result.created_at
    assert result.updated_at >= initial.updated_at
    assert result.endpoint is None
    d.deployments.update_deployment.assert_called_once_with(result, expected_revision=0)


def test_generated_identity_is_distinct_from_model_version(dependencies):
    one = deploy(dependencies, deployment_id=None)
    two = deploy(dependencies, deployment_id=None)
    assert one.id != two.id
    assert one.id not in (one.model_name, one.version)


@pytest.mark.parametrize("overrides", [
    {"model_name": " "}, {"version": ""}, {"deployment_id": ""},
    {"replicas": -1}, {"replicas": True}, {"resources": {"cpu": 1}},
])
def test_invalid_request_performs_no_persistence_or_runtime_io(dependencies, overrides):
    d = dependencies
    with pytest.raises(DomainValidationError):
        deploy(d, **overrides)
    d.models.get_model.assert_not_called()
    d.deployments.save_deployment.assert_not_called()
    d.runtimes.get_runtime.assert_not_called()


@pytest.mark.parametrize("operation,error", [
    ("get_model", ModelNotFoundError("missing model")),
    ("get_version", ModelVersionNotFoundError("missing version")),
])
def test_missing_registered_identity_prevents_deployment(dependencies, operation, error):
    d = dependencies
    getattr(d.models, operation).side_effect = error
    with pytest.raises(type(error)) as captured:
        deploy(d)
    assert captured.value is error
    d.deployments.save_deployment.assert_not_called()
    d.runtimes.get_runtime.assert_not_called()


def test_invalid_model_version_is_ineligible(dependencies):
    d = dependencies
    d.models.get_version.return_value.status = ModelVersionStatus.INVALID
    with pytest.raises(DomainValidationError):
        deploy(d)
    d.deployments.save_deployment.assert_not_called()


@pytest.mark.parametrize("error", [DuplicateDeploymentError("duplicate"), DeploymentPersistenceError("disk")])
def test_initial_save_failure_does_not_load(dependencies, error):
    d = dependencies
    d.deployments.save_deployment.side_effect = error
    with pytest.raises(type(error)) as captured:
        deploy(d)
    assert captured.value is error
    d.runtimes.get_runtime.assert_not_called()


@pytest.mark.parametrize("source,error", [
    ("runtime", ModelLoadingError("bad artifact")),
    ("health", DeploymentHealthCheckError("not ready")),
])
def test_load_or_readiness_failure_records_failed_and_preserves_error(dependencies, source, error):
    d = dependencies
    if source == "runtime":
        d.runtimes.get_runtime.side_effect = error
    else:
        d.checker.check.side_effect = error
    with pytest.raises(type(error)) as captured:
        deploy(d)
    assert captured.value is error
    failed = d.deployments.update_deployment.call_args.args[0]
    assert failed.status is DeploymentStatus.FAILED
    assert d.deployments.save_deployment.call_args.args[0].status is DeploymentStatus.DEPLOYING


def test_failed_state_persistence_exposes_both_errors(dependencies):
    d = dependencies
    operation = ModelLoadingError("bad bytes")
    persistence = DeploymentPersistenceError("disk full")
    d.runtimes.get_runtime.side_effect = operation
    d.deployments.update_deployment.side_effect = persistence
    with pytest.raises(DeploymentRecoveryError) as captured:
        deploy(d)
    assert captured.value.operation_error is operation
    assert captured.value.persistence_error is persistence
    assert captured.value.deployment_id == "deployment-1"
    assert captured.value.__cause__ is persistence


def test_running_write_failure_attempts_failed_from_original_revision(dependencies):
    d = dependencies
    failure = DeploymentPersistenceError("cannot commit RUNNING")
    updates = []

    def update(item, expected_revision):
        updates.append((item.status, expected_revision))
        if item.status is DeploymentStatus.RUNNING:
            raise failure
        return DeploymentSnapshot(item, 1)

    d.deployments.update_deployment.side_effect = update
    with pytest.raises(DeploymentPersistenceError) as captured:
        deploy(d)
    assert captured.value is failure
    assert updates == [(DeploymentStatus.RUNNING, 0), (DeploymentStatus.FAILED, 0)]


def test_stale_activation_is_not_retried_or_overwritten(dependencies):
    d = dependencies
    d.deployments.update_deployment.side_effect = DeploymentConflictError("stopped elsewhere")
    with pytest.raises(DeploymentConflictError):
        deploy(d)
    assert d.deployments.update_deployment.call_count == 1


def test_scaling_preserves_original_snapshot_and_uses_revision(dependencies):
    d = dependencies
    original = snapshot(d)
    scaled = d.manager.scale("deployment-1", 3)
    assert scaled.replicas == 3
    assert scaled.status is DeploymentStatus.RUNNING
    assert scaled.created_at == original.deployment.created_at
    assert scaled.updated_at >= original.deployment.updated_at
    assert original.deployment.replicas == 1
    d.deployments.update_deployment.assert_called_once_with(scaled, expected_revision=4)
    d.runtimes.get_runtime.assert_not_called()


@pytest.mark.parametrize("status", [DeploymentStatus.FAILED, DeploymentStatus.STOPPED])
def test_illegal_scale_does_not_write(dependencies, status):
    d = dependencies
    snapshot(d, status)
    with pytest.raises(InvalidDeploymentStateError):
        d.manager.scale("deployment-1", 3)
    d.deployments.update_deployment.assert_not_called()


@pytest.mark.parametrize("replicas", [-1, True, 1.5])
def test_invalid_scale_does_not_write_or_mutate_snapshot(dependencies, replicas):
    d = dependencies
    original = snapshot(d)
    with pytest.raises(DomainValidationError):
        d.manager.scale("deployment-1", replicas)
    assert original.deployment.replicas == 1
    d.deployments.update_deployment.assert_not_called()


@pytest.mark.parametrize("status", [DeploymentStatus.DEPLOYING, DeploymentStatus.RUNNING, DeploymentStatus.FAILED])
def test_undeploy_preserves_history_and_shared_runtime(dependencies, status):
    d = dependencies
    snapshot(d, status)
    stopped = d.manager.undeploy("deployment-1")
    assert stopped.status is DeploymentStatus.STOPPED
    d.runtimes.invalidate.assert_not_called()
    d.runtimes.clear.assert_not_called()


def test_repeated_undeploy_is_rejected(dependencies):
    d = dependencies
    snapshot(d, DeploymentStatus.STOPPED)
    with pytest.raises(InvalidDeploymentStateError):
        d.manager.undeploy("deployment-1")
    d.deployments.update_deployment.assert_not_called()


@pytest.mark.parametrize("status", [DeploymentStatus.DEPLOYING, DeploymentStatus.FAILED, DeploymentStatus.STOPPED])
def test_nonrunning_deployment_is_not_ready_and_does_not_load(dependencies, status):
    d = dependencies
    snapshot(d, status)
    health = d.manager.check_health("deployment-1")
    assert not health.ready
    assert health.status is status
    d.runtimes.get_runtime.assert_not_called()


def test_running_health_revalidates_current_runtime_without_persisting_state(dependencies):
    d = dependencies
    original = snapshot(d)
    health = d.manager.check_health("deployment-1", smoke_inputs=[[1]])
    assert health.ready
    assert health.kind == "load"
    assert health.status is DeploymentStatus.RUNNING
    d.checker.check.assert_called_once_with(d.runtimes.get_runtime.return_value, smoke_inputs=[[1]])
    d.deployments.update_deployment.assert_not_called()
    assert original.deployment.status is DeploymentStatus.RUNNING


def test_health_failure_marks_running_failed(dependencies):
    d = dependencies
    original = snapshot(d)
    d.runtimes.get_runtime.side_effect = ModelLoadingError("cannot reload evicted runtime")
    with pytest.raises(ModelLoadingError):
        d.manager.check_health("deployment-1")
    updated = d.deployments.update_deployment.call_args.args[0]
    assert updated.status is DeploymentStatus.FAILED
    assert original.deployment.status is DeploymentStatus.RUNNING


def test_health_success_on_stale_snapshot_does_not_report_ready(dependencies):
    d = dependencies
    initial = snapshot(d)
    newer = replace(initial.deployment)
    newer.stop()
    d.deployments.get_deployment.side_effect = [initial, DeploymentSnapshot(newer, 5)]
    with pytest.raises(DeploymentConflictError):
        d.manager.check_health("deployment-1")
    d.deployments.update_deployment.assert_not_called()


def test_history_queries_return_domain_entities(dependencies):
    d = dependencies
    item = snapshot(d)
    d.deployments.list_deployments.return_value = [item]
    assert d.manager.list_deployments("model") == [item.deployment]
    assert d.manager.get_deployment("deployment-1") == item.deployment
    d.deployments.list_deployments.assert_called_once_with("model")


def test_running_and_failed_writes_both_failing_preserves_running_write_error(dependencies):
    d = dependencies
    running_error = DeploymentPersistenceError("RUNNING write failed")
    failed_error = DeploymentPersistenceError("FAILED write failed")
    d.deployments.update_deployment.side_effect = [running_error, failed_error]
    with pytest.raises(DeploymentRecoveryError) as captured:
        deploy(d)
    assert captured.value.operation_error is running_error
    assert captured.value.persistence_error is failed_error
    assert d.deployments.update_deployment.call_count == 2
