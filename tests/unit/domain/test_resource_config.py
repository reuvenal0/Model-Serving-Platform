from dataclasses import FrozenInstanceError

import pytest

from app.core.exceptions import DomainValidationError
from app.domain.resource_config import ResourceConfig


def test_positive_resources():
    resources = ResourceConfig(cpu=0.5, memory_mb=128)
    assert resources.cpu == 0.5
    assert resources.memory_mb == 128


@pytest.mark.parametrize("cpu", [0, -1, float("nan"), float("inf"), -float("inf"), True, "1"])
def test_invalid_cpu(cpu):
    with pytest.raises(DomainValidationError):
        ResourceConfig(cpu=cpu, memory_mb=128)


@pytest.mark.parametrize("memory", [0, -1, 1.5, True, "128"])
def test_invalid_memory(memory):
    with pytest.raises(DomainValidationError):
        ResourceConfig(cpu=1, memory_mb=memory)


def test_resources_are_immutable():
    resources = ResourceConfig(1, 128)
    with pytest.raises(FrozenInstanceError):
        resources.cpu = 2
