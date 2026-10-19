from datetime import datetime, timezone

import pytest

from app.core.exceptions import DomainValidationError
from app.domain.routing_config import RoutingConfig


@pytest.mark.parametrize("weights", [{"v1": 100}, {"v1": 25, "v2": 75}, {"v1": 0, "v2": 100}])
def test_valid_routing(weights):
    config = RoutingConfig("classifier", weights)
    assert config.weights == weights
    assert config.updated_at.tzinfo is timezone.utc


@pytest.mark.parametrize("weights", [{}, {"": 100}, {" ": 100}, {"v1": -1, "v2": 101}, {"v1": 101}, {"v1": 100.0}, {"v1": True, "v2": 99}, {"v1": "100"}, {"v1": 99}, {"v1": 40, "v2": 40}])
def test_invalid_routing(weights):
    with pytest.raises(DomainValidationError):
        RoutingConfig("classifier", weights)


@pytest.mark.parametrize("name", ["", " "])
def test_empty_model_name(name):
    with pytest.raises(DomainValidationError):
        RoutingConfig(name, {"v1": 100})


def test_caller_dictionary_is_copied():
    weights = {"v1": 100}
    config = RoutingConfig("classifier", weights)
    weights["v1"] = 0
    assert config.weights == {"v1": 100}


def test_naive_timestamp_is_rejected():
    with pytest.raises(DomainValidationError):
        RoutingConfig("classifier", {"v1": 100}, datetime(2026, 1, 1))
