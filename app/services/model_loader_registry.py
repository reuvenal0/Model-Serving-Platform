"""Explicit, dependency-injected framework-to-loader lookup."""

from collections.abc import Mapping

from app.core.exceptions import UnsupportedFrameworkError
from app.domain.enums import ModelFramework
from app.interfaces.model_loader import ModelLoader


class ModelLoaderRegistry:
    def __init__(self, loaders: Mapping[ModelFramework, ModelLoader]) -> None:
        self._loaders = dict(loaders)
        for framework in self._loaders:
            if not isinstance(framework, ModelFramework):
                raise UnsupportedFrameworkError(f"Unsupported framework: {framework!r}")

    def get_loader(self, framework: ModelFramework) -> ModelLoader:
        if not isinstance(framework, ModelFramework):
            raise UnsupportedFrameworkError(f"Unsupported framework: {framework!r}")
        try:
            return self._loaders[framework]
        except KeyError as error:
            raise UnsupportedFrameworkError(f"No loader configured for {framework.value}") from error
