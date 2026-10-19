from datetime import datetime, timezone

import pytest

from app.core.exceptions import DomainValidationError
from app.domain.inference_log import InferenceLog


def make_log(**overrides):
    values = dict(request_id="request-1", model_name="classifier", version="v1", latency_ms=0, success=True)
    return InferenceLog(**(values | overrides))


def test_successful_request_metadata():
    log = make_log(input_hash="abc123")
    assert log.latency_ms == 0
    assert log.success is True
    assert log.error_message is None
    assert log.input_hash == "abc123"
    assert log.timestamp.tzinfo is timezone.utc


def test_failed_request_metadata():
    log = make_log(latency_ms=2.5, success=False, error_message="Invalid features")
    assert log.success is False
    assert log.error_message == "Invalid features"


@pytest.mark.parametrize("field", ["request_id", "model_name", "version"])
@pytest.mark.parametrize("value", ["", " "])
def test_empty_identifiers(field, value):
    with pytest.raises(DomainValidationError):
        make_log(**{field: value})


@pytest.mark.parametrize("latency", [-1, float("nan"), float("inf"), True])
def test_invalid_latency(latency):
    with pytest.raises(DomainValidationError):
        make_log(latency_ms=latency)


def test_success_must_be_boolean():
    with pytest.raises(DomainValidationError):
        make_log(success="true")


def test_naive_timestamp_is_rejected():
    with pytest.raises(DomainValidationError):
        make_log(timestamp=datetime(2026, 1, 1))
