from datetime import datetime, timezone

import pytest

from app.core.exceptions import DomainValidationError
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
