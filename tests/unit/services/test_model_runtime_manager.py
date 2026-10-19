from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, FrozenInstanceError
from threading import Barrier, Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.core.exceptions import (
    ArtifactNotFoundError, DomainValidationError, ModelLoadingError,
    ModelVersionNotFoundError, RepositoryError, UnsupportedFrameworkError,
)
from app.domain.enums import ModelFramework, ModelVersionStatus
from app.domain.model_version import ModelVersion
from app.interfaces.artifact_storage import ArtifactStorage
from app.interfaces.model_loader import ModelLoader
from app.interfaces.model_repository import ModelRepository
from app.services.model_loader_registry import ModelLoaderRegistry
from app.services.model_runtime_manager import ModelRuntimeManager


@dataclass
class Runtime:
    artifact: bytes

    def predict(self, inputs):
        return inputs


@pytest.fixture
def dependencies():
    repository = Mock(spec=ModelRepository)
    repository.get_version.side_effect = lambda name, version: ModelVersion(
        name, version, f"{name}/{version}", ModelFramework.ONNX,
    )
    storage = Mock(spec=ArtifactStorage)
    storage.retrieve.side_effect = lambda key: key.encode()
    loader = Mock(spec=ModelLoader)
    loader.load.side_effect = Runtime
    loaders = ModelLoaderRegistry({ModelFramework.ONNX: loader})
    manager = ModelRuntimeManager(repository, storage, loaders, max_cache_size=2)
    return SimpleNamespace(repository=repository, storage=storage, loader=loader,
                           loaders=loaders, manager=manager)


@pytest.fixture
def waiter_joined(monkeypatch):
    """Observe a follower waiting on the private load future without sleeps."""
    joined = Event()

    class ObservableFuture(Future):
        def result(self, timeout=None):
            joined.set()
            return super().result(timeout)

    monkeypatch.setattr("app.services.model_runtime_manager.Future", ObservableFuture)
    return joined


def test_lazy_miss_then_hit_reuses_runtime(dependencies):
    d = dependencies
    d.repository.get_version.assert_not_called()
    first = d.manager.get_runtime("model", "v1")
    assert d.manager.get_runtime("model", "v1") is first
    d.repository.get_version.assert_called_once_with("model", "v1")
    d.storage.retrieve.assert_called_once_with("model/v1")
    d.loader.load.assert_called_once_with(b"model/v1")
    info = d.manager.cache_info()
    assert (info.hits, info.misses, info.loads, info.waits) == (1, 1, 1, 0)
    assert (info.current_size, info.max_size, info.in_flight) == (1, 2, 0)
    with pytest.raises(FrozenInstanceError):
        info.hits = 99


def test_lru_eviction_promotes_hits(dependencies):
    manager = dependencies.manager
    first = manager.get_runtime("model", "v1")
    second = manager.get_runtime("model", "v2")
    assert manager.get_runtime("model", "v1") is first
    manager.get_runtime("model", "v3")
    assert manager.get_runtime("model", "v1") is first
    assert manager.get_runtime("model", "v2") is not second
    info = manager.cache_info()
    assert info.evictions == 2
    assert info.current_size == 2
    assert info.loads == 4
    assert second.predict("still usable after eviction") == "still usable after eviction"


def test_capacity_one_and_distinct_model_version_pairs(dependencies):
    d = dependencies
    manager = ModelRuntimeManager(d.repository, d.storage, d.loaders, max_cache_size=1)
    for name, version in [("one", "v1"), ("one", "v2"), ("two", "v2")]:
        assert manager.get_runtime(name, version).artifact == f"{name}/{version}".encode()
        assert manager.cache_info().current_size == 1
    assert manager.cache_info().evictions == 2


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "2", None])
def test_capacity_must_be_a_positive_integer(dependencies, capacity):
    d = dependencies
    with pytest.raises(DomainValidationError):
        ModelRuntimeManager(d.repository, d.storage, d.loaders, max_cache_size=capacity)


@pytest.mark.parametrize("method", ["get_runtime", "invalidate"])
@pytest.mark.parametrize("name,version", [("", "v1"), ("model", " ")])
def test_invalid_identifiers_do_not_access_backends(dependencies, method, name, version):
    with pytest.raises(DomainValidationError):
        getattr(dependencies.manager, method)(name, version)
    dependencies.repository.get_version.assert_not_called()


def test_invalidation_and_clear_preserve_existing_callers_and_lifetime_statistics(dependencies):
    manager = dependencies.manager
    first = manager.get_runtime("model", "v1")
    manager.invalidate("model", "v1")
    manager.invalidate("missing", "v1")
    assert manager.cache_info().current_size == 0
    assert manager.get_runtime("model", "v1") is not first
    manager.get_runtime("model", "v2")
    manager.clear()
    manager.clear()
    assert manager.cache_info().current_size == 0
    assert manager.cache_info().loads == 3
    assert first.predict("still usable") == "still usable"


def test_failed_load_is_not_cached_and_can_be_retried(dependencies):
    d = dependencies
    failure = ModelLoadingError("bad artifact")
    recovered = Runtime(b"recovered")
    d.loader.load.side_effect = [failure, recovered]
    with pytest.raises(ModelLoadingError) as captured:
        d.manager.get_runtime("model", "v1")
    assert captured.value is failure
    assert d.manager.cache_info().current_size == 0
    assert d.manager.cache_info().in_flight == 0
    assert d.manager.get_runtime("model", "v1") is recovered
    assert d.manager.cache_info().load_failures == 1
    assert d.manager.cache_info().loads == 2


@pytest.mark.parametrize("failure", [ModelVersionNotFoundError("missing"), RepositoryError("unavailable")])
def test_repository_errors_are_preserved(dependencies, failure):
    d = dependencies
    d.repository.get_version.side_effect = failure
    with pytest.raises(type(failure)) as captured:
        d.manager.get_runtime("model", "v1")
    assert captured.value is failure
    d.storage.retrieve.assert_not_called()
    assert d.manager.cache_info().load_failures == 1


def test_missing_artifact_is_preserved(dependencies):
    d = dependencies
    failure = ArtifactNotFoundError("missing bytes")
    d.storage.retrieve.side_effect = failure
    with pytest.raises(ArtifactNotFoundError) as captured:
        d.manager.get_runtime("model", "v1")
    assert captured.value is failure
    d.loader.load.assert_not_called()
    assert d.manager.cache_info().current_size == 0


def test_invalid_version_is_not_loaded_or_changed(dependencies):
    d = dependencies
    item = ModelVersion("model", "v1", "key", ModelFramework.ONNX, status=ModelVersionStatus.INVALID)
    d.repository.get_version.side_effect = None
    d.repository.get_version.return_value = item
    with pytest.raises(ModelLoadingError, match="REGISTERED"):
        d.manager.get_runtime("model", "v1")
    assert item.status is ModelVersionStatus.INVALID
    d.storage.retrieve.assert_not_called()


def test_unsupported_framework_loader_is_explicit(dependencies):
    d = dependencies
    manager = ModelRuntimeManager(d.repository, d.storage, ModelLoaderRegistry({}))
    with pytest.raises(UnsupportedFrameworkError):
        manager.get_runtime("model", "v1")
    assert manager.cache_info().current_size == 0


def test_invalid_loader_result_is_not_cached(dependencies):
    dependencies.loader.load.side_effect = None
    dependencies.loader.load.return_value = object()
    with pytest.raises(ModelLoadingError, match="predict"):
        dependencies.manager.get_runtime("model", "v1")
    assert dependencies.manager.cache_info().current_size == 0


def test_same_key_concurrent_requests_share_one_load(dependencies, waiter_joined):
    d = dependencies
    started, release = Event(), Event()
    runtime = Runtime(b"shared")

    def load(artifact):
        started.set()
        assert release.wait(5), "Test did not release load"
        return runtime

    d.loader.load.side_effect = load
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(d.manager.get_runtime, "model", "v1")
        assert started.wait(5)
        second = pool.submit(d.manager.get_runtime, "model", "v1")
        try:
            assert waiter_joined.wait(5)
            info = d.manager.cache_info()
            assert (info.loads, info.waits, info.in_flight) == (1, 1, 1)
        finally:
            release.set()
        assert first.result(timeout=5) is runtime
        assert second.result(timeout=5) is runtime
    d.loader.load.assert_called_once()
    assert d.manager.get_runtime("model", "v1") is runtime


def test_different_keys_load_concurrently(dependencies):
    d = dependencies
    together = Barrier(2)

    def load(artifact):
        together.wait(timeout=5)
        return Runtime(artifact)

    d.loader.load.side_effect = load
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(d.manager.get_runtime, "model", "v1")
        second = pool.submit(d.manager.get_runtime, "model", "v2")
        assert first.result(timeout=10).artifact == b"model/v1"
        assert second.result(timeout=10).artifact == b"model/v2"
    assert d.manager.cache_info().current_size == 2


def test_shared_failure_reaches_waiters_and_next_request_retries(dependencies, waiter_joined):
    d = dependencies
    started, release = Event(), Event()
    failure = ModelLoadingError("simulated load failure")

    def fail(artifact):
        started.set()
        assert release.wait(5)
        raise failure

    d.loader.load.side_effect = fail
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(d.manager.get_runtime, "model", "v1")
        assert started.wait(5)
        second = pool.submit(d.manager.get_runtime, "model", "v1")
        try:
            assert waiter_joined.wait(5)
        finally:
            release.set()
        for result in [first, second]:
            with pytest.raises(ModelLoadingError) as captured:
                result.result(timeout=5)
            assert captured.value is failure
    assert d.manager.cache_info().in_flight == 0
    assert d.manager.cache_info().load_failures == 1
    d.loader.load.side_effect = Runtime
    d.manager.get_runtime("model", "v1")
    assert d.loader.load.call_count == 2


@pytest.mark.parametrize("operation", ["invalidate", "clear"])
def test_invalidation_during_load_never_resurrects_entry(dependencies, waiter_joined, operation):
    d = dependencies
    started, release = Event(), Event()
    old = Runtime(b"old")

    def load(artifact):
        started.set()
        assert release.wait(5)
        return old

    d.loader.load.side_effect = load
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(d.manager.get_runtime, "model", "v1")
        assert started.wait(5)
        waiter = pool.submit(d.manager.get_runtime, "model", "v1")
        try:
            assert waiter_joined.wait(5)
            if operation == "invalidate":
                d.manager.invalidate("model", "v1")
            else:
                d.manager.clear()
            assert d.manager.cache_info().in_flight == 0
        finally:
            release.set()
        assert first.result(timeout=5) is old
        assert waiter.result(timeout=5) is old
    assert d.manager.cache_info().current_size == 0
    d.loader.load.side_effect = Runtime
    assert d.manager.get_runtime("model", "v1") is not old


@pytest.mark.parametrize("operation", ["invalidate", "clear"])
@pytest.mark.parametrize("old_fails", [False, True])
def test_detached_completion_cannot_replace_new_generation(dependencies, operation, old_fails):
    d = dependencies
    started, release = Event(), Event()
    old, new = Runtime(b"old"), Runtime(b"new")

    def load(artifact):
        if not started.is_set():
            started.set()
            assert release.wait(5)
            if old_fails:
                raise ModelLoadingError("old generation failed")
            return old
        return new

    d.loader.load.side_effect = load
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(d.manager.get_runtime, "model", "v1")
        assert started.wait(5)
        try:
            if operation == "invalidate":
                d.manager.invalidate("model", "v1")
            else:
                d.manager.clear()
            assert d.manager.get_runtime("model", "v1") is new
        finally:
            release.set()
        if old_fails:
            with pytest.raises(ModelLoadingError):
                first.result(timeout=5)
        else:
            assert first.result(timeout=5) is old
    assert d.manager.get_runtime("model", "v1") is new
    assert d.manager.cache_info().current_size == 1
    assert d.manager.cache_info().in_flight == 0
