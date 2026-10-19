import pickle
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from app.core.exceptions import ModelLoadingError, PredictionError, UntrustedArtifactError
from app.infrastructure.execution.sklearn_model_loader import SklearnModelLoader


def test_real_estimator_predictions_are_reused(sklearn_artifact):
    loader = SklearnModelLoader(trusted=True)
    with patch("app.infrastructure.execution.sklearn_model_loader.pickle.loads", wraps=pickle.loads) as loads:
        runtime = loader.load(sklearn_artifact)
        np.testing.assert_allclose(runtime.predict([[3.0], [4.0]]), [7.0, 9.0])
        np.testing.assert_allclose(runtime.predict(np.array([[5.0]])), [11.0])
        loads.assert_called_once_with(sklearn_artifact)


@pytest.mark.parametrize("artifact", [b"", b"not-a-pickle", b"\x80\x05", "not bytes"])
def test_corrupted_or_invalid_artifact(artifact):
    with pytest.raises(ModelLoadingError):
        SklearnModelLoader(trusted=True).load(artifact)


@pytest.mark.parametrize("value", [{"not": "an estimator"}, LinearRegression(), StandardScaler()])
def test_rejects_unusable_estimator(value):
    with pytest.raises(ModelLoadingError) as captured:
        SklearnModelLoader(trusted=True).load(pickle.dumps(value))
    assert captured.value.__cause__ is not None


def test_trust_required_before_any_deserialization(sklearn_artifact):
    with patch("app.infrastructure.execution.sklearn_model_loader.pickle.loads") as loads:
        for loader in [SklearnModelLoader(), SklearnModelLoader(trusted=False)]:
            with pytest.raises(UntrustedArtifactError):
                loader.load(sklearn_artifact)
        loads.assert_not_called()


@pytest.mark.parametrize("inputs", [[1.0, 2.0], [[1.0, 2.0]], [["not-a-number"]], None])
def test_invalid_prediction_inputs_preserve_backend_cause(sklearn_artifact, inputs):
    runtime = SklearnModelLoader(trusted=True).load(sklearn_artifact)
    with pytest.raises(PredictionError) as captured:
        runtime.predict(inputs)
    assert captured.value.__cause__ is not None
    np.testing.assert_allclose(runtime.predict([[3.0]]), [7.0])
