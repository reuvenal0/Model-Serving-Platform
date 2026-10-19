from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import pytest

from app.core.exceptions import DomainValidationError, InvalidDeploymentStateError
from app.domain.deployment import Deployment
from app.domain.enums import DeploymentStatus, ModelVersionStatus
from app.domain.resource_config import ResourceConfig


def make_deployment(**overrides):
    values = dict(id="deployment-1", model_name="classifier", version="v1", replicas=1, resources=ResourceConfig(1, 256))
    return Deployment(**(values | overrides))


@pytest.mark.parametrize("status", list(DeploymentStatus))
@pytest.mark.parametrize("replicas", [0, 2])
def test_valid_deployment(status, replicas):
    deployment = make_deployment(status=status, replicas=replicas)
    assert deployment.status is status
    assert deployment.replicas == replicas
    assert deployment.endpoint is None
    assert deployment.created_at.tzinfo is timezone.utc
    assert deployment.updated_at.tzinfo is timezone.utc
    assert DeploymentStatus(status.value) is status


@pytest.mark.parametrize("field", ["id", "model_name", "version"])
@pytest.mark.parametrize("value", ["", " "])
def test_empty_identifiers(field, value):
    with pytest.raises(DomainValidationError):
        make_deployment(**{field: value})


@pytest.mark.parametrize("replicas", [-1, 1.5, True])
def test_invalid_replicas(replicas):
    with pytest.raises(DomainValidationError):
        make_deployment(replicas=replicas)


@pytest.mark.parametrize("status", ["running", ModelVersionStatus.REGISTERED])
def test_invalid_status(status):
    with pytest.raises(DomainValidationError):
        make_deployment(status=status)


def test_resources_must_be_a_value_object():
    with pytest.raises(DomainValidationError):
        make_deployment(resources={"cpu": 1, "memory_mb": 128})


@pytest.mark.parametrize("field", ["created_at", "updated_at"])
def test_naive_timestamps_are_rejected(field):
    with pytest.raises(DomainValidationError):
        make_deployment(**{field: datetime(2026, 1, 1)})


TRANSITIONS = [
    ("activate", DeploymentStatus.RUNNING, {DeploymentStatus.DEPLOYING}),
    ("fail", DeploymentStatus.FAILED, {DeploymentStatus.DEPLOYING, DeploymentStatus.RUNNING}),
    ("stop", DeploymentStatus.STOPPED, {
        DeploymentStatus.DEPLOYING, DeploymentStatus.RUNNING, DeploymentStatus.FAILED,
    }),
]


@pytest.mark.parametrize("initial", list(DeploymentStatus))
@pytest.mark.parametrize("operation,target,allowed", TRANSITIONS, ids=["activate", "fail", "stop"])
def test_complete_transition_matrix(initial, operation, target, allowed, monkeypatch):
    before = datetime(2026, 1, 1, tzinfo=timezone.utc)
    after = before + timedelta(seconds=10)
    deployment = make_deployment(status=initial, created_at=before, updated_at=before)
    monkeypatch.setattr("app.domain.deployment.utc_now", lambda: after)
    if initial in allowed:
        assert getattr(deployment, operation)() is None
        assert deployment.status is target
        assert deployment.updated_at == after
        assert deployment.updated_at.tzinfo is timezone.utc
    else:
        with pytest.raises(InvalidDeploymentStateError):
            getattr(deployment, operation)()
        assert deployment.status is initial
        assert deployment.updated_at == before
    assert deployment.created_at == before
    assert deployment.replicas == 1
    assert deployment.resources == ResourceConfig(1, 256)


@pytest.mark.parametrize("status", [DeploymentStatus.DEPLOYING, DeploymentStatus.RUNNING])
@pytest.mark.parametrize("replicas", [0, 1, 5])
def test_scale_updates_replicas_and_timestamp(status, replicas, monkeypatch):
    before = datetime(2026, 1, 1, tzinfo=timezone.utc)
    after = before + timedelta(seconds=10)
    deployment = make_deployment(status=status, created_at=before, updated_at=before)
    monkeypatch.setattr("app.domain.deployment.utc_now", lambda: after)
    assert deployment.scale(replicas) is None
    assert deployment.replicas == replicas
    assert deployment.status is status
    assert deployment.updated_at == after
    assert deployment.updated_at.tzinfo is timezone.utc
    assert deployment.created_at == before


@pytest.mark.parametrize("status", [DeploymentStatus.FAILED, DeploymentStatus.STOPPED])
def test_scale_rejected_for_inactive_deployments(status):
    deployment = make_deployment(status=status)
    before = deployment.updated_at
    with pytest.raises(InvalidDeploymentStateError):
        deployment.scale(3)
    assert deployment.status is status
    assert deployment.replicas == 1
    assert deployment.updated_at == before


@pytest.mark.parametrize("replicas", [-1, 1.5, True, "2", None])
@pytest.mark.parametrize("status", list(DeploymentStatus))
def test_invalid_scale_does_not_mutate_entity(replicas, status):
    deployment = make_deployment(status=status)
    before = deployment.updated_at
    with pytest.raises(DomainValidationError):
        deployment.scale(replicas)
    assert deployment.status is status
    assert deployment.replicas == 1
    assert deployment.updated_at == before


@pytest.mark.parametrize("field,value", [
    ("id", ""), ("model_name", " "), ("version", ""), ("replicas", -1),
    ("status", DeploymentStatus.STOPPED), ("resources", None),
    ("endpoint", "http://example.test"), ("created_at", datetime(2026, 1, 1)),
    ("updated_at", datetime(2026, 1, 1)),
])
def test_public_assignment_cannot_bypass_domain_operations(field, value):
    deployment = make_deployment()
    original = getattr(deployment, field)
    with pytest.raises(FrozenInstanceError):
        setattr(deployment, field, value)
    assert getattr(deployment, field) == original
    with pytest.raises(FrozenInstanceError):
        delattr(deployment, field)


def test_deployment_lifecycle():
    deployment = make_deployment()
    deployment.scale(2)
    deployment.activate()
    deployment.scale(3)
    deployment.fail()
    deployment.stop()
    assert deployment.status is DeploymentStatus.STOPPED
    assert deployment.replicas == 3
    with pytest.raises(InvalidDeploymentStateError):
        deployment.activate()


def test_non_utc_timestamps_are_still_normalized():
    local = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=2)))
    deployment = make_deployment(created_at=local, updated_at=local)
    assert deployment.created_at == local
    assert deployment.created_at.hour == 10
    assert deployment.created_at.tzinfo is timezone.utc
    assert deployment.updated_at.tzinfo is timezone.utc


def test_validated_reconstruction_preserves_constructor_api():
    original = Deployment("d-1", "classifier", "v1", 1, ResourceConfig(1, 256))
    restored = replace(original, status=DeploymentStatus.RUNNING, endpoint="http://example.test")
    assert restored.status is DeploymentStatus.RUNNING
    assert restored.endpoint == "http://example.test"
    assert restored.created_at == original.created_at
    assert original.status is DeploymentStatus.DEPLOYING
    with pytest.raises(DomainValidationError):
        replace(original, replicas=-1)


def test_mutable_deployment_is_not_hashable():
    with pytest.raises(TypeError):
        hash(make_deployment())
