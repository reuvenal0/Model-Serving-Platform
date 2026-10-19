from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

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


@pytest.mark.parametrize("weights", [{"v2": 100}, {"v1": 30, "v2": 70}, {"v1": 0, "v2": 100}])
def test_update_replaces_entire_configuration(weights, monkeypatch):
    before = datetime(2026, 1, 1, tzinfo=timezone.utc)
    after = before + timedelta(seconds=10)
    config = RoutingConfig("classifier", {"old": 100}, before)
    monkeypatch.setattr("app.domain.routing_config.utc_now", lambda: after)
    assert config.update_weights(weights) is None
    assert config.weights == weights
    assert "old" not in config.weights
    assert config.model_name == "classifier"
    assert config.updated_at == after
    assert config.updated_at.tzinfo is timezone.utc


@pytest.mark.parametrize("weights", [
    {}, {"": 100}, {" ": 100}, {"v1": -1, "v2": 101}, {"v1": 101},
    {"v1": 100.0}, {"v1": True, "v2": 99}, {"v1": "100"},
    {"v1": 99}, {"v1": 40, "v2": 40}, None, [],
])
def test_invalid_replacement_preserves_weights_and_timestamp(weights):
    before = datetime(2026, 1, 1, tzinfo=timezone.utc)
    config = RoutingConfig("classifier", {"v1": 100}, before)
    snapshot = config.weights
    with pytest.raises(DomainValidationError):
        config.update_weights(weights)
    assert config.weights is snapshot
    assert config.weights == {"v1": 100}
    assert config.updated_at == before


def test_update_defensively_copies_caller_mapping():
    config = RoutingConfig("classifier", {"v1": 100})
    replacement = {"v2": 100}
    config.update_weights(replacement)
    replacement.clear()
    assert config.weights == {"v2": 100}


def test_exposed_weights_are_read_only():
    config = RoutingConfig("classifier", {"v1": 100})
    with pytest.raises(TypeError):
        config.weights["v1"] = 0
    with pytest.raises(TypeError):
        del config.weights["v1"]
    assert config.weights == {"v1": 100}


def test_previous_weight_view_remains_a_valid_snapshot_after_update():
    config = RoutingConfig("classifier", {"v1": 100})
    before = config.weights
    config.update_weights({"v2": 100})
    assert before == {"v1": 100}
    assert config.weights == {"v2": 100}


@pytest.mark.parametrize("field,value", [
    ("weights", {"invalid": 99}), ("model_name", ""),
    ("updated_at", datetime(2026, 1, 1)),
])
def test_direct_assignment_and_deletion_are_blocked(field, value):
    config = RoutingConfig("classifier", {"v1": 100})
    original = getattr(config, field)
    with pytest.raises(FrozenInstanceError):
        setattr(config, field, value)
    assert getattr(config, field) == original
    with pytest.raises(FrozenInstanceError):
        delattr(config, field)


def test_mapping_can_be_exported_as_an_independent_dictionary():
    config = RoutingConfig("classifier", {"v1": 100})
    exported = dict(config.weights)
    exported["v1"] = 0
    assert config.weights == {"v1": 100}


def test_validated_reconstruction_accepts_read_only_mapping():
    original = RoutingConfig("classifier", {"v1": 100})
    clone = replace(original)
    assert clone == original
    original.update_weights({"v2": 100})
    assert clone.weights == {"v1": 100}
    with pytest.raises(DomainValidationError):
        replace(original, weights={"v1": 99})


def test_non_utc_timestamp_is_normalized():
    local = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=2)))
    config = RoutingConfig("classifier", {"v1": 100}, local)
    assert config.updated_at == local
    assert config.updated_at.hour == 10
    assert config.updated_at.tzinfo is timezone.utc


def test_mutable_routing_config_is_not_hashable():
    with pytest.raises(TypeError):
        hash(RoutingConfig("classifier", {"v1": 100}))
