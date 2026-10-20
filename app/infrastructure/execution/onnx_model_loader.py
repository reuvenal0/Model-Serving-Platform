"""CPU ONNX execution with a deliberately narrow dense-tensor contract."""

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
import onnxruntime as ort

from app.core.exceptions import ModelLoadingError, PredictionError, PredictionInputError
from app.interfaces.model_loader import ModelLoader


_DTYPES = {
    "tensor(float)": np.dtype("float32"),
    "tensor(double)": np.dtype("float64"),
    "tensor(int32)": np.dtype("int32"),
    "tensor(int64)": np.dtype("int64"),
    "tensor(bool)": np.dtype("bool"),
}


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    type: str
    shape: tuple[int | str | None, ...]


def _metadata(nodes) -> tuple[TensorMetadata, ...]:
    result = []
    for node in nodes:
        if node.type not in _DTYPES or node.shape is None:
            raise ModelLoadingError(f"Unsupported ONNX tensor metadata for {node.name!r}: {node.type}")
        result.append(TensorMetadata(node.name, node.type, tuple(node.shape)))
    return tuple(result)


class OnnxLoadedModel:
    def __init__(self, session: ort.InferenceSession) -> None:
        self._session = session
        self._inputs = _metadata(session.get_inputs())
        self._outputs = _metadata(session.get_outputs())
        if not self.inputs or not self.outputs:
            raise ModelLoadingError("ONNX model must expose at least one input and output")
        self._prediction_lock = Lock()

    @property
    def inputs(self) -> tuple[TensorMetadata, ...]:
        return self._inputs

    @property
    def outputs(self) -> tuple[TensorMetadata, ...]:
        return self._outputs

    @staticmethod
    def _tensor(value: Any, metadata: TensorMetadata) -> np.ndarray:
        dtype = _DTYPES[metadata.type]
        if isinstance(value, np.ndarray):
            if value.dtype != dtype:
                raise PredictionInputError(f"Input {metadata.name!r} requires dtype {dtype}; received {value.dtype}")
            return value
        if not isinstance(value, (list, tuple)):
            raise PredictionInputError(f"Input {metadata.name!r} must be a NumPy array or nested list/tuple")
        array = np.asarray(value)
        allowed_kinds = "b" if dtype.kind == "b" else "iu" if dtype.kind in "iu" else "iuf"
        if array.dtype.kind not in allowed_kinds:
            raise PredictionInputError(f"Input {metadata.name!r} cannot be converted from {array.dtype} to {dtype}")
        if dtype.kind in "iu" and array.size:
            limits = np.iinfo(dtype)
            if array.min().item() < limits.min or array.max().item() > limits.max:
                raise PredictionInputError(f"Input {metadata.name!r} exceeds the range of {dtype}")
        with np.errstate(over="raise", invalid="raise"):
            return array.astype(dtype)

    def predict(self, inputs: Any) -> dict[str, np.ndarray]:
        """Accept a named tensor mapping, or raw data for a single input.

        Arrays must have exactly the declared dtype. Numeric Python sequences
        are converted (without float-to-int truncation or string coercion).
        Ranks, fixed dimensions, and shared symbolic dimensions are checked.
        All declared inputs are required, and extra inputs are rejected.
        Return a dictionary of output name to NumPy array.
        """
        try:
            if isinstance(inputs, Mapping):
                values = dict(inputs)
            elif len(self.inputs) == 1:
                values = {self.inputs[0].name: inputs}
            else:
                raise PredictionInputError("Multiple ONNX inputs require a mapping by input name")
            expected = {item.name for item in self.inputs}
            if set(values) != expected:
                raise PredictionInputError(
                    f"ONNX input names differ; missing={expected - set(values)!r}, "
                    f"unexpected={set(values) - expected!r}"
                )
            feed = {}
            symbols: dict[str, int] = {}
            for metadata in self.inputs:
                array = self._tensor(values[metadata.name], metadata)
                if array.ndim != len(metadata.shape):
                    raise PredictionInputError(f"Input {metadata.name!r} requires rank {len(metadata.shape)}")
                for actual, dimension in zip(array.shape, metadata.shape):
                    if isinstance(dimension, int) and actual != dimension:
                        raise PredictionInputError(f"Input {metadata.name!r} requires dimension {dimension}; received {actual}")
                    if isinstance(dimension, str):
                        if dimension in symbols and symbols[dimension] != actual:
                            raise PredictionInputError(f"Conflicting sizes for symbolic dimension {dimension!r}")
                        symbols[dimension] = actual
                feed[metadata.name] = array
            names = [item.name for item in self.outputs]
            with self._prediction_lock:
                predictions = self._session.run(names, feed)
            return dict(zip(names, predictions))
        except PredictionError:
            raise
        except (ValueError, TypeError, FloatingPointError, OverflowError) as error:
            raise PredictionInputError("ONNX prediction input is invalid") from error
        except Exception as error:
            raise PredictionError("ONNX prediction failed") from error


class OnnxModelLoader(ModelLoader):
    """Load a self-contained ONNX graph from trusted bytes using the CPU provider.

    Only dense float32/float64/int32/int64/bool inputs and outputs with known
    rank are supported. External tensor files, custom operators, optional inputs,
    sequences, maps, strings, and sparse tensors are outside this contract.
    ONNX execution also runs in-process and is not a security sandbox.
    """

    def load(self, artifact: bytes) -> OnnxLoadedModel:
        if not isinstance(artifact, bytes) or not artifact:
            raise ModelLoadingError("ONNX artifact must be nonempty bytes")
        try:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                artifact, sess_options=options, providers=["CPUExecutionProvider"],
            )
            return OnnxLoadedModel(session)
        except ModelLoadingError:
            raise
        except Exception as error:
            raise ModelLoadingError("Could not load ONNX model") from error
