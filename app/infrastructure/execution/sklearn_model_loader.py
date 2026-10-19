"""Trusted Pickle loading of fitted sklearn estimators; not a sandbox."""

import pickle
from threading import Lock
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from app.core.exceptions import ModelLoadingError, PredictionError, UntrustedArtifactError
from app.interfaces.model_loader import ModelLoader


class SklearnLoadedModel:
    """Reuse one estimator; delegate sklearn input validation to predict()."""

    def __init__(self, estimator: BaseEstimator) -> None:
        self._estimator = estimator
        self._prediction_lock = Lock()

    def predict(self, inputs: Any) -> Any:
        """Accept estimator-compatible features (usually 2D lists/ndarrays).

        Return the estimator's native output, typically a NumPy array. Calls on
        this runtime are serialized because estimator thread safety varies.
        """
        try:
            with self._prediction_lock:
                return self._estimator.predict(inputs)
        except Exception as error:
            raise PredictionError("Scikit-learn prediction failed") from error


class SklearnModelLoader(ModelLoader):
    """Load standard Pickle bytes ONLY from an explicitly trusted source.

    Pickle (and joblib) can execute arbitrary code during deserialization. The
    trust flag is caller acknowledgement, not provenance verification or a
    security check. Do not enable it for an untrusted registry/storage source.
    Use matching training/serving dependencies. Compressed joblib is not supported.
    """

    def __init__(self, *, trusted: bool = False) -> None:
        self._trusted = trusted is True

    def load(self, artifact: bytes) -> SklearnLoadedModel:
        if not self._trusted:
            raise UntrustedArtifactError("Explicitly trusted=True is required for Pickle loading")
        if not isinstance(artifact, bytes) or not artifact:
            raise ModelLoadingError("Scikit-learn artifact must be nonempty bytes")
        try:
            # Trust must be established before this call; validation afterwards
            # cannot undo code executed by Pickle.
            estimator = pickle.loads(artifact)
            if not isinstance(estimator, BaseEstimator) or not callable(getattr(estimator, "predict", None)):
                raise TypeError("Artifact must contain an sklearn estimator with predict()")
            check_is_fitted(estimator)
        except Exception as error:
            raise ModelLoadingError("Could not load fitted scikit-learn estimator") from error
        return SklearnLoadedModel(estimator)
