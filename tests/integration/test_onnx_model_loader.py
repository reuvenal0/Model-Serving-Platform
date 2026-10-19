from unittest.mock import patch

import numpy as np
from onnx import TensorProto, helper
import pytest

from app.core.exceptions import ModelLoadingError, PredictionError
from app.infrastructure.execution.onnx_model_loader import OnnxModelLoader


def identity_artifact(data_type, shape):
    graph = helper.make_graph(
        [helper.make_node("Identity", ["X"], ["Y"])], "identity",
        [helper.make_tensor_value_info("X", data_type, shape)],
        [helper.make_tensor_value_info("Y", data_type, shape)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8).SerializeToString()


def test_real_onnx_predictions_reuse_session_and_expose_metadata(onnx_artifact):
    import onnxruntime as ort

    with patch("app.infrastructure.execution.onnx_model_loader.ort.InferenceSession", wraps=ort.InferenceSession) as sessions:
        runtime = OnnxModelLoader().load(onnx_artifact)
        assert runtime.inputs[0].name == "X"
        assert runtime.inputs[0].type == "tensor(float)"
        assert runtime.inputs[0].shape == ("batch", 2)
        assert runtime.outputs[0].name == "Y"
        np.testing.assert_allclose(runtime.predict([[1, 2], [3, 4]])["Y"], [[3, 5], [7, 9]])
        np.testing.assert_allclose(runtime.predict({"X": np.array([[5, 6]], dtype=np.float32)})["Y"], [[11, 13]])
        sessions.assert_called_once()


@pytest.mark.parametrize("artifact", [b"", b"invalid-onnx", b"\x01\xff", "not bytes"])
def test_invalid_artifacts(artifact):
    with pytest.raises(ModelLoadingError):
        OnnxModelLoader().load(artifact)


@pytest.mark.parametrize("inputs", [
    {}, {"wrong": [[1, 2]]}, {"X": [[1, 2]], "extra": [[1, 2]]},
    [[1, 2, 3]], [1, 2], [[[1, 2]]], [["1", "2"]],
    [[True, False]], [[1], [2, 3]], None,
    np.array([[1, 2]], dtype=np.float64), np.array([[1, 2]], dtype=np.int64),
])
def test_invalid_prediction_inputs(onnx_artifact, inputs):
    runtime = OnnxModelLoader().load(onnx_artifact)
    with pytest.raises(PredictionError):
        runtime.predict(inputs)
    np.testing.assert_allclose(runtime.predict([[1, 2]])["Y"], [[3, 5]])


def test_backend_prediction_errors_are_chained(onnx_artifact):
    runtime = OnnxModelLoader().load(onnx_artifact)
    with patch.object(runtime._session, "run", side_effect=RuntimeError("native execution failed")):
        with pytest.raises(PredictionError) as captured:
            runtime.predict([[1, 2]])
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_multiple_inputs_require_names_and_matching_symbolic_dimensions(onnx_two_input_artifact):
    runtime = OnnxModelLoader().load(onnx_two_input_artifact)
    np.testing.assert_allclose(runtime.predict({"left": [[1, 2]], "right": [[3, 4]]})["sum"], [[4, 6]])
    for inputs in [[[1, 2]], {"left": [[1, 2]]},
                   {"left": [[1, 2]], "right": [[1, 2], [3, 4]]}]:
        with pytest.raises(PredictionError):
            runtime.predict(inputs)


@pytest.mark.parametrize("data_type,dtype,values", [
    (TensorProto.FLOAT, "float32", [[1.5, 2.5]]),
    (TensorProto.DOUBLE, "float64", [[1.5, 2.5]]),
    (TensorProto.INT32, "int32", [[1, 2]]),
    (TensorProto.INT64, "int64", [[1, 2]]),
    (TensorProto.BOOL, "bool", [[True, False]]),
])
def test_supported_dense_types(data_type, dtype, values):
    runtime = OnnxModelLoader().load(identity_artifact(data_type, [1, 2]))
    result = runtime.predict(values)["Y"]
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, np.array(values, dtype=dtype))


@pytest.mark.parametrize("values", [[[1.5, 2.5]], [[2**40, 1]], [[True, False]]])
def test_integer_inputs_do_not_truncate_overflow_or_accept_booleans(values):
    runtime = OnnxModelLoader().load(identity_artifact(TensorProto.INT32, [1, 2]))
    with pytest.raises(PredictionError):
        runtime.predict(values)


def test_floating_conversion_overflow_is_explicit():
    runtime = OnnxModelLoader().load(identity_artifact(TensorProto.FLOAT, [1, 2]))
    with pytest.raises(PredictionError):
        runtime.predict([[1e100, 2]])


def test_unsupported_tensor_type_is_rejected_at_load():
    with pytest.raises(ModelLoadingError, match="Unsupported ONNX"):
        OnnxModelLoader().load(identity_artifact(TensorProto.STRING, [1, 2]))


def test_rank_zero_tensor_and_dynamic_batch(onnx_artifact):
    scalar = OnnxModelLoader().load(identity_artifact(TensorProto.FLOAT, []))
    np.testing.assert_array_equal(scalar.predict(np.array(3, dtype=np.float32))["Y"], np.array(3, dtype=np.float32))
    runtime = OnnxModelLoader().load(onnx_artifact)
    assert runtime.predict(np.empty((0, 2), dtype=np.float32))["Y"].shape == (0, 2)
