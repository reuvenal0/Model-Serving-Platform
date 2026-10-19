from unittest.mock import Mock

import pytest

from app.core.exceptions import UnsupportedFrameworkError
from app.domain.enums import ModelFramework
from app.interfaces.model_loader import ModelLoader
from app.services.model_loader_registry import ModelLoaderRegistry


@pytest.mark.parametrize("framework", list(ModelFramework))
def test_injected_loader_lookup(framework):
    loader = Mock(spec=ModelLoader)
    configured = {framework: loader}
    registry = ModelLoaderRegistry(configured)
    configured.clear()
    assert registry.get_loader(framework) is loader


@pytest.mark.parametrize("framework", ["sklearn", "onnx", "tensorflow", None])
def test_non_enum_framework_is_explicitly_rejected(framework):
    registry = ModelLoaderRegistry({ModelFramework.SKLEARN: Mock(spec=ModelLoader)})
    with pytest.raises(UnsupportedFrameworkError):
        registry.get_loader(framework)


def test_missing_loader_is_explicit():
    with pytest.raises(UnsupportedFrameworkError):
        ModelLoaderRegistry({}).get_loader(ModelFramework.ONNX)


def test_invalid_mapping_key_is_rejected():
    with pytest.raises(UnsupportedFrameworkError):
        ModelLoaderRegistry({"sklearn": Mock(spec=ModelLoader)})
