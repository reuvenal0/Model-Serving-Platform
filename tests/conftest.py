"""Small local model fixtures; no external artifacts or downloads."""

import pickle

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest
from sklearn.linear_model import LinearRegression


@pytest.fixture
def sklearn_artifact():
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    estimator = LinearRegression().fit(features, np.array([1.0, 3.0, 5.0, 7.0]))
    return pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)


@pytest.fixture
def onnx_artifact():
    """Compute Y = 2*X + 1 for a dynamic batch of two float32 features."""
    graph = helper.make_graph(
        [helper.make_node("Mul", ["X", "two"], ["scaled"]),
         helper.make_node("Add", ["scaled", "one"], ["Y"])],
        "affine", [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 2])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 2])],
        initializer=[numpy_helper.from_array(np.array(2, dtype=np.float32), "two"),
                     numpy_helper.from_array(np.array(1, dtype=np.float32), "one")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    return model.SerializeToString()


@pytest.fixture
def onnx_two_input_artifact():
    graph = helper.make_graph(
        [helper.make_node("Add", ["left", "right"], ["sum"])], "sum",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", 2])
         for name in ["left", "right"]],
        [helper.make_tensor_value_info("sum", TensorProto.FLOAT, ["batch", 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    return model.SerializeToString()
