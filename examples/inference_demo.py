"""Trusted, in-process sklearn execution; run after installing .[dev]."""

from dataclasses import asdict
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory

import numpy as np
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from sklearn.linear_model import LinearRegression

from app.config import Settings
from app.domain.enums import ModelFramework
from app.infrastructure.database.database import create_sqlite_engine, initialize_database
from app.infrastructure.database.sqlite_model_repository import SQLiteModelRepository
from app.infrastructure.execution.onnx_model_loader import OnnxModelLoader
from app.infrastructure.execution.sklearn_model_loader import SklearnModelLoader
from app.infrastructure.storage.local_artifact_storage import LocalArtifactStorage
from app.services.model_loader_registry import ModelLoaderRegistry
from app.services.model_registry import ModelRegistry
from app.services.model_runtime_manager import ModelRuntimeManager


def onnx_demo_artifact() -> bytes:
    # ONNX graph-building is a development dependency, not needed by the loader.
    from onnx import TensorProto, checker, helper, numpy_helper

    graph = helper.make_graph(
        [helper.make_node("Mul", ["X", "two"], ["scaled"]),
         helper.make_node("Add", ["scaled", "one"], ["Y"])],
        "affine", [helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 1])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 1])],
        initializer=[numpy_helper.from_array(np.array(2, dtype=np.float32), "two"),
                     numpy_helper.from_array(np.array(1, dtype=np.float32), "one")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    checker.check_model(model)
    return model.SerializeToString()


def main() -> None:
    # This estimator is trained and serialized here; its provenance is trusted.
    estimator = LinearRegression().fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0])
    artifact = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
    with TemporaryDirectory(prefix="inference-demo-") as directory:
        root = Path(directory)
        settings = Settings(
            database_url=URL.create("sqlite", database=str(root / "metadata.db")).render_as_string(),
            artifact_storage_dir=root / "artifacts",
        )
        engine = create_sqlite_engine(settings.database_url)
        try:
            initialize_database(engine)
            repository = SQLiteModelRepository(sessionmaker(engine))
            storage = LocalArtifactStorage(settings.artifact_storage_dir)
            registry = ModelRegistry(repository, storage)
            loaders = ModelLoaderRegistry({
                ModelFramework.SKLEARN: SklearnModelLoader(trusted=True),
                ModelFramework.ONNX: OnnxModelLoader(),
            })
            manager = ModelRuntimeManager(
                repository, storage, loaders, max_cache_size=settings.runtime_cache_max_size,
            )
            registry.register_model("linear-demo", "Locally trusted model: y = 2*x + 1")
            registry.register_version("linear-demo", "v1", ModelFramework.SKLEARN, artifact)
            first = manager.get_runtime("linear-demo", "v1")
            first_prediction = first.predict([[3.0], [4.0]])
            np.testing.assert_allclose(first_prediction, [7.0, 9.0])
            second = manager.get_runtime("linear-demo", "v1")
            second_prediction = second.predict([[5.0]])
            np.testing.assert_allclose(second_prediction, [11.0])
            if second is not first:
                raise RuntimeError("Runtime cache did not reuse the sklearn model")
            print("First sklearn prediction:", first_prediction.round(6).tolist())
            print("Second sklearn prediction:", second_prediction.round(6).tolist())
            print("Same cached runtime:", first is second)
            print("Cache after sklearn:", asdict(manager.cache_info()))

            registry.register_version("linear-demo", "onnx-v1", ModelFramework.ONNX, onnx_demo_artifact())
            onnx_runtime = manager.get_runtime("linear-demo", "onnx-v1")
            onnx_prediction = onnx_runtime.predict([[3.0], [4.0]])["Y"]
            np.testing.assert_allclose(onnx_prediction, [[7.0], [9.0]])
            print("ONNX prediction:", onnx_prediction.tolist())
            print("Final cache:", asdict(manager.cache_info()))
            manager.clear()
            print("Inference demo passed; temporary metadata and artifacts will be removed.")
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
