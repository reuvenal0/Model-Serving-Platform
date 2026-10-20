"""Bounded artifact reads, backend races, runtime defenses, and RSS probes."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from app.config import Settings
from app.core.exceptions import ArtifactStorageError, ArtifactTooLargeError, DomainValidationError
from app.domain.enums import ModelFramework
from app.domain.model_version import ModelVersion
from app.infrastructure.resources import process_memory
from app.infrastructure.storage.local_artifact_storage import LocalArtifactStorage
from app.interfaces.artifact_storage import ArtifactStorage
from app.interfaces.model_loader import ModelLoader
from app.interfaces.model_repository import ModelRepository
from app.services.model_loader_registry import ModelLoaderRegistry
from app.services.model_runtime_manager import ModelRuntimeManager
from app.services.runtime_resource_monitor import RuntimeResourceMonitor


@pytest.mark.parametrize('size', [3,4])
def test_local_under_and_exact_limit(tmp_path, size):
    storage = LocalArtifactStorage(tmp_path)
    storage.save('key', b'x'*size)
    assert storage.retrieve_limited('key', max_bytes=4) == b'x'*size


def test_oversize_rejected_before_any_file_read(tmp_path, monkeypatch):
    storage = LocalArtifactStorage(tmp_path)
    storage.save('key',b'large')
    original = Path.open
    reads = Mock(side_effect=AssertionError('oversized artifact was read'))

    def open_file(path, *args, **kwargs):
        stream = original(path,*args,**kwargs)
        wrapped = MagicMock(wraps=stream)
        wrapped.__enter__.return_value = wrapped
        wrapped.__exit__.side_effect = lambda *ignored: stream.close()
        wrapped.read = reads
        return wrapped
    with monkeypatch.context() as patch:
        patch.setattr(Path,'open',open_file)
        with pytest.raises(ArtifactTooLargeError):
            storage.retrieve_limited('key',max_bytes=4)
    reads.assert_not_called()
    assert storage.retrieve('key') == b'large'


def test_inaccurate_file_size_cannot_bypass_bounded_read(tmp_path, monkeypatch):
    storage = LocalArtifactStorage(tmp_path)
    storage.save('key',b'x'*100)
    monkeypatch.setattr('app.infrastructure.storage.local_artifact_storage.os.fstat',
                        lambda descriptor: SimpleNamespace(st_size=0))
    with pytest.raises(ArtifactTooLargeError):
        storage.retrieve_limited('key',max_bytes=4)
    assert storage.retrieve('key') == b'x'*100


def test_growth_after_size_check_is_rejected(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    import os

    storage = LocalArtifactStorage(tmp_path)
    storage.save('key',b'data')
    checked, release = Event(), Event()
    original = os.fstat

    def paused(descriptor):
        result = original(descriptor)
        checked.set()
        assert release.wait(5)
        return result
    monkeypatch.setattr('app.infrastructure.storage.local_artifact_storage.os.fstat',paused)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(storage.retrieve_limited,'key',max_bytes=4)
        assert checked.wait(5)
        try:
            with storage._path('key').open('ab') as stream:
                stream.write(b'!')
        finally:
            release.set()
        with pytest.raises(ArtifactTooLargeError):
            pending.result(timeout=5)
    assert storage.retrieve('key') == b'data!'


@pytest.mark.parametrize('framework',list(ModelFramework))
@pytest.mark.parametrize('size', [3,4,5])
def test_runtime_independently_checks_backend_bytes(framework, size):
    models = Mock(spec=ModelRepository)
    models.get_version.return_value = ModelVersion('model','v1','key',framework)
    storage = Mock(spec=ArtifactStorage)
    storage.retrieve_limited.return_value = b'x'*size  # Faulty backend ignores its limit.
    loader = Mock(spec=ModelLoader)
    loader.load.return_value = Mock(predict=Mock())
    manager = ModelRuntimeManager(models,storage,ModelLoaderRegistry({framework:loader}),max_artifact_bytes=4)
    if size > 4:
        with pytest.raises(ArtifactTooLargeError):
            manager.get_runtime('model','v1')
        loader.load.assert_not_called()
        assert manager.cache_info().current_size == 0
        assert manager.cache_info().load_failures == 1
    else:
        assert manager.get_runtime('model','v1') is loader.load.return_value
        loader.load.assert_called_once_with(b'x'*size)
    storage.retrieve_limited.assert_called_once_with('key',max_bytes=4)


@pytest.mark.parametrize('limit',[0,-1,True,1.5,None])
def test_runtime_limit_must_be_positive_integer(limit):
    with pytest.raises(DomainValidationError):
        ModelRuntimeManager(Mock(),Mock(),Mock(),max_artifact_bytes=limit)


@pytest.mark.parametrize('limit',[0,-1,True,1.5])
def test_settings_limit_validation(limit):
    with pytest.raises(ValueError):
        Settings(max_runtime_artifact_bytes=limit)


def test_settings_limit_default_and_environment(monkeypatch):
    monkeypatch.delenv('MAX_RUNTIME_ARTIFACT_BYTES',raising=False)
    assert Settings().max_runtime_artifact_bytes == 100*1024*1024
    monkeypatch.setenv('MAX_RUNTIME_ARTIFACT_BYTES','32')
    assert Settings().max_runtime_artifact_bytes == 32
    monkeypatch.setenv('MAX_RUNTIME_ARTIFACT_BYTES','0')
    with pytest.raises(ValueError):
        Settings()


def test_legacy_storage_adapter_remains_compatible():
    class Legacy(ArtifactStorage):
        def save(self,key,content): pass
        def retrieve(self,key): return b'data'
        def delete(self,key): pass
        def exists(self,key): return True
    assert Legacy().retrieve_limited('key',max_bytes=4) == b'data'
    with pytest.raises(ArtifactTooLargeError):
        Legacy().retrieve_limited('key',max_bytes=3)


def test_resource_snapshot_does_not_claim_per_model_memory():
    manager = Mock(spec=ModelRuntimeManager)
    manager.cache_info.return_value = 'cache snapshot'
    snapshot = RuntimeResourceMonitor(manager,lambda: 1234).snapshot()
    assert snapshot.rss_available and snapshot.rss_bytes == 1234
    assert snapshot.cache == 'cache snapshot'
    assert RuntimeResourceMonitor(manager,lambda: None).snapshot().rss_available is False


def test_native_process_rss_is_available_on_supported_host():
    import sys
    rss = process_memory.process_rss_bytes()
    if sys.platform == 'win32' or sys.platform.startswith('linux'):
        assert type(rss) is int and rss >= 0
    else:
        assert rss is None


def test_linux_rss_uses_current_resident_pages(monkeypatch):
    monkeypatch.setattr(process_memory.Path,'read_text',lambda *args,**kwargs: '100 3 0 0 0 0 0')
    monkeypatch.setattr(process_memory.os,'sysconf',lambda name: 4096,raising=False)
    assert process_memory._linux_rss_bytes() == 12288


@pytest.mark.parametrize('platform',['unsupported','darwin'])
def test_unsupported_platform_reports_unavailable(monkeypatch, platform):
    monkeypatch.setattr(process_memory.sys,'platform',platform)
    assert process_memory.process_rss_bytes() is None


def test_probe_failure_reports_unavailable(monkeypatch):
    monkeypatch.setattr(process_memory.sys,'platform','linux')
    monkeypatch.setattr(process_memory,'_linux_rss_bytes',Mock(side_effect=OSError('private path')))
    assert process_memory.process_rss_bytes() is None


@pytest.mark.parametrize('content',[None,'not bytes',bytearray(b'data')])
def test_runtime_rejects_invalid_backend_content(content):
    models = Mock(spec=ModelRepository)
    models.get_version.return_value = ModelVersion('model','v1','key',ModelFramework.ONNX)
    storage = Mock(spec=ArtifactStorage)
    storage.retrieve_limited.return_value = content
    loader = Mock(spec=ModelLoader)
    manager = ModelRuntimeManager(models,storage,ModelLoaderRegistry({ModelFramework.ONNX:loader}))
    with pytest.raises(ArtifactStorageError):
        manager.get_runtime('model','v1')
    loader.load.assert_not_called()
