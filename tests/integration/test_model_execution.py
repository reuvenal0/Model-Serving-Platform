from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from app.core.exceptions import ModelVersionNotFoundError
from app.domain.enums import ModelFramework
from app.infrastructure.execution.onnx_model_loader import OnnxModelLoader
from app.infrastructure.execution.sklearn_model_loader import SklearnModelLoader
from app.infrastructure.storage.local_artifact_storage import LocalArtifactStorage
from app.services.model_loader_registry import ModelLoaderRegistry
from app.services.model_registry import ModelRegistry
from app.services.model_runtime_manager import ModelRuntimeManager


@pytest.mark.parametrize("framework,fixture_name,inputs,expected", [
    (ModelFramework.SKLEARN, "sklearn_artifact", [[3.0], [4.0]], [7.0, 9.0]),
    (ModelFramework.ONNX, "onnx_artifact", [[1.0, 2.0]], [[3.0, 5.0]]),
])
def test_registry_to_runtime_prediction_and_reuse(
    repository, tmp_path, request, framework, fixture_name, inputs, expected,
):
    artifact = request.getfixturevalue(fixture_name)
    storage = LocalArtifactStorage(tmp_path / "artifacts")
    registry = ModelRegistry(repository, storage)
    model = registry.register_model("real-model")
    version = registry.register_version("real-model", "v1", framework, artifact)
    loaders = ModelLoaderRegistry({
        ModelFramework.SKLEARN: SklearnModelLoader(trusted=True),
        ModelFramework.ONNX: OnnxModelLoader(),
    })
    manager = ModelRuntimeManager(repository, storage, loaders, max_cache_size=1)
    first = manager.get_runtime("real-model", "v1")
    prediction = first.predict(inputs)
    actual = prediction["Y"] if framework is ModelFramework.ONNX else prediction
    np.testing.assert_allclose(actual, expected)
    assert manager.get_runtime("real-model", "v1") is first
    assert (manager.cache_info().misses, manager.cache_info().hits) == (1, 1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(first.predict, [inputs] * 8))
    for result in results:
        np.testing.assert_allclose(result["Y"] if framework is ModelFramework.ONNX else result, expected)
    assert registry.get_model("real-model") == model
    assert registry.get_version("real-model", "v1") == version
    assert registry.download_artifact("real-model", "v1") == artifact
    manager.invalidate("real-model", "v1")
    assert manager.get_runtime("real-model", "v1") is not first
    registry.delete_version("real-model", "v1")
    # Registry semantics are unchanged: cached runtimes require explicit invalidation.
    assert manager.get_runtime("real-model", "v1").predict(inputs) is not None
    manager.invalidate("real-model", "v1")
    with pytest.raises(ModelVersionNotFoundError):
        manager.get_runtime("real-model", "v1")
