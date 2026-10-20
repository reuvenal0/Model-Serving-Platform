"""Lazy model runtimes with an LRU cache and per-key shared loading futures."""

from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Lock

from app.core.exceptions import ArtifactNotFoundError, ArtifactStorageError, ArtifactTooLargeError, DomainValidationError, ModelLoadingError
from app.domain._validation import nonempty
from app.domain.enums import ModelVersionStatus
from app.interfaces.artifact_storage import ArtifactStorage
from app.interfaces.loaded_model import LoadedModel
from app.interfaces.model_repository import ModelRepository
from app.services.model_loader_registry import ModelLoaderRegistry


@dataclass(frozen=True)
class RuntimeCacheInfo:
    hits: int
    misses: int
    waits: int
    loads: int
    load_failures: int
    evictions: int
    current_size: int
    max_size: int
    in_flight: int


class ModelRuntimeManager:
    """Cache immutable registered versions by (model_name, version).

    The lock protects bookkeeping only. Storage, deserialization, waiting, and
    predictions do not run under it. Concurrent misses share one future per key.
    Invalidation detaches an in-progress future without cancelling its callers;
    those callers still receive its result/error, but it cannot enter the cache.
    Later requests may load a fresh generation concurrently with the detached one.
    Eviction/invalidation only release cache references, never close a runtime in
    use. Immediate reclamation of native/Python memory is not guaranteed.
    """

    def __init__(
        self, repository: ModelRepository, storage: ArtifactStorage,
        loaders: ModelLoaderRegistry, *, max_cache_size: int = 16,
        max_artifact_bytes: int = 100 * 1024 * 1024,
    ) -> None:
        if type(max_cache_size) is not int or max_cache_size <= 0:
            raise DomainValidationError("max_cache_size must be a strictly positive integer")
        if type(max_artifact_bytes) is not int or max_artifact_bytes <= 0:
            raise DomainValidationError("max_artifact_bytes must be a strictly positive integer")
        self._repository = repository
        self._storage = storage
        self._loaders = loaders
        self._max_cache_size = max_cache_size
        self._max_artifact_bytes = max_artifact_bytes
        self._cache: OrderedDict[tuple[str, str], LoadedModel] = OrderedDict()
        self._loading: dict[tuple[str, str], Future[LoadedModel]] = {}
        self._lock = Lock()
        self._hits = self._misses = self._waits = self._loads = 0
        self._failures = self._evictions = 0

    def _load(self, model_name: str, version: str) -> LoadedModel:
        metadata = self._repository.get_version(model_name, version)
        if metadata.status is not ModelVersionStatus.REGISTERED:
            raise ModelLoadingError("Only REGISTERED model versions can be loaded")
        artifact = self._storage.retrieve_limited(metadata.artifact_location, max_bytes=self._max_artifact_bytes)
        # Independent guard: a racing or faulty backend must not bypass the cap.
        if not isinstance(artifact, bytes):
            raise ArtifactStorageError("Artifact backend did not return bytes")
        if len(artifact) > self._max_artifact_bytes:
            raise ArtifactTooLargeError("Artifact exceeds maximum runtime loading size")
        loader = self._loaders.get_loader(metadata.framework)
        runtime = loader.load(artifact)
        if not callable(getattr(runtime, "predict", None)):
            raise ModelLoadingError("Loader returned a runtime without predict()")
        return runtime

    def verify_available(self, model_name: str, version: str) -> None:
        """Recovery preflight even for a cached runtime; no deserialization.

        This checks registry state and artifact existence, not artifact integrity
        or prediction accuracy. Actual cache misses use bounded load-readiness.
        """
        self._repository.get_model(model_name)
        metadata = self._repository.get_version(model_name, version)
        if metadata.status is not ModelVersionStatus.REGISTERED:
            raise ModelLoadingError("Only REGISTERED model versions can be loaded")
        if not self._storage.exists(metadata.artifact_location):
            raise ArtifactNotFoundError("Runtime artifact does not exist")

    def get_runtime(self, model_name: str, version: str) -> LoadedModel:
        nonempty(model_name, "model_name")
        nonempty(version, "version")
        key = (model_name, version)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            self._misses += 1
            future = self._loading.get(key)
            owner = future is None
            if owner:
                future = Future()
                self._loading[key] = future
                self._loads += 1
            else:
                self._waits += 1
        if not owner:
            return future.result()
        try:
            runtime = self._load(model_name, version)
        except BaseException as error:
            # Even interruption must release waiters; the exception is re-raised.
            with self._lock:
                self._failures += 1
                if self._loading.get(key) is future:
                    del self._loading[key]
            future.set_exception(error)
            raise
        evicted = None
        with self._lock:
            if self._loading.get(key) is future:
                del self._loading[key]
                self._cache[key] = runtime
                if len(self._cache) > self._max_cache_size:
                    evicted = self._cache.popitem(last=False)
                    self._evictions += 1
        # Complete outside the lock; a future callback could execute caller code.
        # Keep an evicted runtime alive until after leaving the bookkeeping lock.
        future.set_result(runtime)
        return runtime

    def invalidate(self, model_name: str, version: str) -> None:
        nonempty(model_name, "model_name")
        nonempty(version, "version")
        with self._lock:
            removed = self._cache.pop((model_name, version), None)
            detached = self._loading.pop((model_name, version), None)
        # Local references release outside the lock; existing callers remain valid.

    def clear(self) -> None:
        with self._lock:
            previous_cache, previous_loads = self._cache, self._loading
            self._cache = OrderedDict()
            self._loading = {}
        # Pending callers continue; detached futures cannot repopulate the cache.

    def cache_info(self) -> RuntimeCacheInfo:
        """Lifetime counters; clear() does not reset them.

        Misses include waiters, loads count leaders, and in_flight counts only
        currently tracked generations (detached work can still be running).
        """
        with self._lock:
            return RuntimeCacheInfo(
                self._hits, self._misses, self._waits, self._loads,
                self._failures, self._evictions, len(self._cache),
                self._max_cache_size, len(self._loading),
            )
