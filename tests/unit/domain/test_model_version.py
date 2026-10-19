from datetime import datetime, timezone

import pytest

from app.core.exceptions import DomainValidationError
from app.domain.enums import ModelFramework, ModelVersionStatus
from app.domain.model_version import ModelVersion


def make_version(**overrides):
    values = dict(model_name="classifier", version="v1", artifact_location="artifacts/v1.pkl", framework=ModelFramework.SKLEARN)
    return ModelVersion(**(values | overrides))


@pytest.mark.parametrize("framework", list(ModelFramework))
@pytest.mark.parametrize("status", list(ModelVersionStatus))
def test_valid_version(framework, status):
    version = make_version(framework=framework, status=status)
    assert version.framework is framework
    assert version.status is status
    assert version.created_at.tzinfo is timezone.utc


@pytest.mark.parametrize("field", ["model_name", "version", "artifact_location"])
@pytest.mark.parametrize("value", ["", " "])
def test_empty_identifiers(field, value):
    with pytest.raises(DomainValidationError):
        make_version(**{field: value})


@pytest.mark.parametrize("field,value", [("framework", "sklearn"), ("status", "registered")])
def test_enums_are_required(field, value):
    with pytest.raises(DomainValidationError):
        make_version(**{field: value})


def test_schema_defaults_are_independent():
    first, second = make_version(), make_version()
    first.input_schema["features"] = {"type": "array"}
    first.output_schema["prediction"] = {"type": "number"}
    assert second.input_schema == {}
    assert second.output_schema == {}
    assert first.input_schema is not first.output_schema


def test_naive_timestamp_is_rejected():
    with pytest.raises(DomainValidationError):
        make_version(created_at=datetime(2026, 1, 1))


def test_enum_string_values():
    assert ModelFramework("sklearn") is ModelFramework.SKLEARN
    assert ModelFramework("onnx") is ModelFramework.ONNX
    assert ModelVersionStatus("registered") is ModelVersionStatus.REGISTERED
    assert ModelVersionStatus("invalid") is ModelVersionStatus.INVALID
