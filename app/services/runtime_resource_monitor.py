"""Operational resource visibility through injected runtime and process probes."""

from collections.abc import Callable
from dataclasses import dataclass

from app.services.model_runtime_manager import ModelRuntimeManager, RuntimeCacheInfo


@dataclass(frozen=True)
class RuntimeResourceSnapshot:
    rss_bytes: int | None
    rss_available: bool
    cache: RuntimeCacheInfo


class RuntimeResourceMonitor:
    def __init__(self, runtimes: ModelRuntimeManager, rss_reader: Callable[[], int | None]) -> None:
        self._runtimes = runtimes
        self._rss_reader = rss_reader

    def snapshot(self) -> RuntimeResourceSnapshot:
        rss = self._rss_reader()
        if type(rss) is not int or rss < 0:
            rss = None
        return RuntimeResourceSnapshot(rss, rss is not None, self._runtimes.cache_info())
