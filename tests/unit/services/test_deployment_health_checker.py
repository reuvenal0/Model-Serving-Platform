from unittest.mock import Mock

import pytest

from app.core.exceptions import DeploymentHealthCheckError, PredictionError
from app.services.deployment_health_checker import DeploymentHealthChecker


def test_load_readiness_does_not_invent_prediction_input():
    runtime = Mock()
    result = DeploymentHealthChecker().check(runtime)
    assert result.kind == "load"
    assert result.checked_at.utcoffset().total_seconds() == 0
    runtime.predict.assert_not_called()


@pytest.mark.parametrize("inputs", [[[1, 2]], {"X": [[1, 2]]}, None])
def test_explicit_smoke_inputs_are_used_exactly(inputs):
    runtime = Mock()
    result = DeploymentHealthChecker().check(runtime, smoke_inputs=inputs)
    assert result.kind == "smoke"
    runtime.predict.assert_called_once_with(inputs)


@pytest.mark.parametrize("runtime", [None, object()])
def test_absent_or_invalid_runtime_is_not_ready(runtime):
    with pytest.raises(DeploymentHealthCheckError):
        DeploymentHealthChecker().check(runtime)


def test_smoke_failure_preserves_prediction_context():
    error = PredictionError("invalid features")
    runtime = Mock()
    runtime.predict.side_effect = error
    with pytest.raises(DeploymentHealthCheckError) as captured:
        DeploymentHealthChecker().check(runtime, smoke_inputs=[[1]])
    assert captured.value.__cause__ is error
