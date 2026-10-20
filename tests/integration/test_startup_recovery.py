"""Automatic recovery with revision guards, failure isolation and clean ownership."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from unittest.mock import Mock

from tests.api_client import TestClient
import pytest

from app.config import Settings
from app.core.container import ServiceContainer
from app.core import exceptions as errors
from app.domain.deployment import Deployment
from app.domain.enums import DeploymentStatus, ModelFramework
from app.domain.resource_config import ResourceConfig
from app.main import create_app


@pytest.fixture
def settings(tmp_path):
    return Settings(database_url=f"sqlite:///{tmp_path / 'recovery.db'}",
                    artifact_storage_dir=tmp_path / 'artifacts',trust_sklearn_artifacts=True)


def prepare(settings, artifact, versions=('v1',)):
    with ServiceContainer(settings) as container:
        container.model_registry.register_model('model')
        for version in versions:
            container.model_registry.register_version('model',version,ModelFramework.SKLEARN,artifact)
            container.deployment_manager.deploy('model',version,ResourceConfig(1,128),deployment_id=version)
        return container.deployment_repository.get_deployment(versions[0])


def test_owned_restart_recovers_prediction_routing_and_monitoring(settings, sklearn_artifact):
    first_app = create_app(settings)
    with TestClient(first_app) as client:
        container = first_app.state.container
        container.model_registry.register_model('model')
        container.model_registry.register_version('model','v1',ModelFramework.SKLEARN,sklearn_artifact)
        container.deployment_manager.deploy('model','v1',ResourceConfig(1,128),deployment_id='v1')
        container.routing_manager.configure_routing('model',{'v1':100},0)
        assert client.post('/predict/model',json={'inputs':[[4]]}).json()['predictions'] == pytest.approx([9])
        revision = container.deployment_repository.get_deployment('v1').revision
    assert first_app.state.container is None and container._closed
    second_app = create_app(settings)
    with TestClient(second_app) as client:
        report = second_app.state.startup_report
        assert len(report.health) == 1 and report.health[0].ready and not report.failures
        fresh = second_app.state.container
        assert fresh.runtime_manager.cache_info().loads == 1
        assert fresh.deployment_repository.get_deployment('v1').revision == revision
        assert client.get('/deployments/v1/health').json()['ready'] is True
        assert client.post('/predict/model',json={'inputs':[[5]]}).json()['predictions'] == pytest.approx([11])
        assert client.get('/metrics/model').json()['total_requests'] == 2
        assert client.get('/routing/model').json()['weights'] == {'v1':100}
        resources = client.get('/runtime/resources').json()
        assert resources['cache']['current_size'] == 1 and resources['cache']['max_size'] == 16
        assert set(resources) == {'rss_bytes','rss_available','cache'}
    assert fresh._closed and fresh.runtime_manager.cache_info().current_size == 0


def test_interrupted_deploying_becomes_failed(settings, sklearn_artifact):
    prepare(settings,sklearn_artifact)
    with ServiceContainer(settings) as container:
        container.deployment_repository.save_deployment(Deployment('interrupted','model','v1',1,ResourceConfig(1,128)))
    app = create_app(settings)
    with TestClient(app) as client:
        interrupted = app.state.container.deployment_repository.get_deployment('interrupted')
        assert interrupted.deployment.status is DeploymentStatus.FAILED and interrupted.revision == 1
        assert client.get('/deployments/interrupted/health').json()['ready'] is False
        assert client.post('/predict/model',json={'version':'v1','inputs':[[4]]}).status_code == 200


@pytest.mark.parametrize('mode',['missing','untrusted','invalid','corrupt'])
def test_per_deployment_failures_allow_liveness_but_reject_inference(settings, sklearn_artifact, mode, caplog):
    prepare(settings,sklearn_artifact)
    with ServiceContainer(settings) as container:
        metadata = container.model_registry.get_version('model','v1')
        if mode == 'missing':
            container.artifact_storage.delete(metadata.artifact_location)
        elif mode == 'invalid':
            from sqlalchemy import text
            with container.engine.begin() as connection:
                connection.execute(text("UPDATE model_versions SET status='invalid' WHERE version='v1'"))
        elif mode == 'corrupt':
            container.artifact_storage.delete(metadata.artifact_location)
            container.artifact_storage.save(metadata.artifact_location,b'not an estimator')
    selected = replace(settings,trust_sklearn_artifacts=False) if mode == 'untrusted' else settings
    app = create_app(selected)
    with TestClient(app) as client:
        assert len(app.state.startup_report.failures) == 1
        assert client.get('/health').json() == {'status':'ok'}
        assert app.state.container.deployment_manager.get_deployment('v1').status is DeploymentStatus.FAILED
        assert client.post('/predict/model',json={'version':'v1','inputs':[[4]]}).status_code == 503
    diagnostics = [record for record in caplog.records if 'startup_' in record.getMessage()]
    assert diagnostics and all(record.exc_info is None for record in diagnostics)
    assert str(settings.artifact_storage_dir) not in caplog.text


def test_failure_isolation_and_repeated_recovery(settings, sklearn_artifact):
    prepare(settings,sklearn_artifact,('v1','v2'))
    with ServiceContainer(settings) as container:
        key = container.model_registry.get_version('model','v2').artifact_location
        container.artifact_storage.delete(key)
    app = create_app(settings)
    with TestClient(app) as client:
        assert len(app.state.startup_report.failures) == 1
        assert client.post('/predict/model',json={'version':'v1','inputs':[[4]]}).status_code == 200
        assert client.post('/predict/model',json={'version':'v2','inputs':[[4]]}).status_code == 503
        before = app.state.container.deployment_repository.get_deployment('v1')
        repeated = app.state.container.reconcile_startup()
        assert not repeated.failures
        assert app.state.container.deployment_repository.get_deployment('v1') == before
        assert app.state.container.deployment_manager.get_deployment('v2').status is DeploymentStatus.FAILED


def test_recovery_does_not_bypass_lru_capacity(settings, sklearn_artifact):
    prepare(settings,sklearn_artifact,('v1','v2','v3'))
    app = create_app(replace(settings,runtime_cache_max_size=1))
    with TestClient(app) as client:
        assert all(item.ready for item in app.state.startup_report.health)
        manager = app.state.container.runtime_manager
        assert manager.cache_info().current_size == 1 and manager.cache_info().evictions == 2
        assert client.post('/predict/model',json={'version':'v1','inputs':[[4]]}).status_code == 200
        assert manager.cache_info().loads == 4


def test_injected_cached_runtime_does_not_hide_missing_artifact_on_repeated_startup(settings, sklearn_artifact):
    prepare(settings,sklearn_artifact)
    with ServiceContainer(settings) as container:
        app = create_app(container=container)
        with TestClient(app):
            assert app.state.startup_report.health[0].ready
        key = container.model_registry.get_version('model','v1').artifact_location
        container.artifact_storage.delete(key)
        with TestClient(app) as client:
            assert len(app.state.startup_report.failures) == 1
            assert client.post('/predict/model',json={'version':'v1','inputs':[[4]]}).status_code == 503
        assert container._closed is False


def test_revision_conflict_during_startup_never_restores_running(settings, sklearn_artifact, monkeypatch):
    prepare(settings,sklearn_artifact)
    with ServiceContainer(settings) as container:
        original = container.health_checker.check
        def stop(runtime,**kwargs):
            container.deployment_manager.undeploy('v1')
            return original(runtime,**kwargs)
        monkeypatch.setattr(container.health_checker,'check',stop)
        app = create_app(container=container)
        with TestClient(app) as client:
            assert isinstance(app.state.startup_report.failures[0].error,errors.DeploymentConflictError)
            assert container.deployment_manager.get_deployment('v1').status is DeploymentStatus.STOPPED
            assert client.post('/predict/model',json={'version':'v1','inputs':[[4]]}).status_code == 503


def test_fatal_database_listing_failure_aborts_and_closes_owned_container(settings, monkeypatch, caplog):
    real = ServiceContainer
    created = []
    def construct(selected):
        instance = real(selected)
        created.append(instance)
        instance.deployment_repository.list_deployments = Mock(side_effect=errors.RepositoryError('/private/database'))
        return instance
    monkeypatch.setattr('app.main.ServiceContainer',construct)
    app = create_app(settings)
    with pytest.raises(errors.StartupRecoveryError):
        with TestClient(app):
            pytest.fail('startup accepted traffic after infrastructure failure')
    assert created[0]._closed and app.state.container is None
    assert '/private/database' not in caplog.text and 'Traceback' not in caplog.text


def test_unavailable_database_fails_before_serving(settings, tmp_path):
    app = create_app(replace(settings,database_url=f"sqlite:///{tmp_path / 'missing' / 'metadata.db'}"))
    with pytest.raises(errors.StartupRecoveryError):
        with TestClient(app):
            pytest.fail('database initialization failure was ignored')
    assert app.state.container is None


def test_persistence_failure_inside_recovery_is_fatal(settings, sklearn_artifact, monkeypatch):
    prepare(settings,sklearn_artifact)
    with ServiceContainer(settings) as container:
        container.artifact_storage.delete(container.model_registry.get_version('model','v1').artifact_location)
        monkeypatch.setattr(container.deployment_repository,'update_deployment',Mock(side_effect=errors.RepositoryError('unavailable')))
        with pytest.raises(errors.StartupRecoveryError):
            with TestClient(create_app(container=container)):
                pytest.fail('failed recovery persistence was ignored')
        assert container.deployment_manager.get_deployment('v1').status is DeploymentStatus.RUNNING
        # The application did not advertise readiness despite the unrepaired row.


def test_oversize_recovery_preserves_artifact_and_version_metadata(settings, sklearn_artifact):
    prepare(settings,sklearn_artifact)
    with ServiceContainer(settings) as container:
        before = container.model_registry.get_version('model','v1')
        content = container.model_registry.download_artifact('model','v1')
    app = create_app(replace(settings,max_runtime_artifact_bytes=len(content)-1))
    with TestClient(app) as client:
        container = app.state.container
        assert isinstance(app.state.startup_report.failures[0].error,errors.ArtifactTooLargeError)
        assert container.model_registry.get_version('model','v1') == before
        assert container.model_registry.download_artifact('model','v1') == content
        assert client.post('/predict/model',json={'version':'v1','inputs':[[4]]}).status_code == 503


def test_concurrent_startup_sweeps_share_one_inflight_reconciliation(settings, monkeypatch):
    from concurrent.futures import Future
    from app.services.deployment_manager import ReconciliationReport
    started, release, joined = Event(), Event(), Event()
    class ObservedFuture(Future):
        def result(self,timeout=None):
            joined.set()
            return super().result(timeout)
    monkeypatch.setattr('app.core.container.Future',ObservedFuture)
    with ServiceContainer(settings) as container:
        def blocked():
            started.set()
            assert release.wait(5)
            return ReconciliationReport((),())
        reconcile = Mock(side_effect=blocked)
        monkeypatch.setattr(container.deployment_manager,'reconcile',reconcile)
        with ThreadPoolExecutor(max_workers=2) as pool:
            one = pool.submit(container.reconcile_startup)
            assert started.wait(5)
            two = pool.submit(container.reconcile_startup)
            try:
                assert joined.wait(5)
            finally:
                release.set()
            assert one.result(timeout=5) is two.result(timeout=5)
        reconcile.assert_called_once()


def test_failed_recovery_keeps_strict_routing_policy_and_monitoring(settings, sklearn_artifact):
    prepare(settings,sklearn_artifact,('v1','v2'))
    with ServiceContainer(settings) as container:
        container.routing_manager.configure_routing('model',{'v1':90,'v2':10},0)
        container.artifact_storage.delete(container.model_registry.get_version('model','v2').artifact_location)
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.post('/predict/model',json={'inputs':[[4]]}).status_code == 503
        assert client.post('/predict/model',json={'version':'v1','inputs':[[4]]}).status_code == 200
        assert client.get('/routing/model').json()['weights'] == {'v1':90,'v2':10}
        metrics = client.get('/metrics/model').json()
        assert metrics['total_requests'] == 2 and metrics['failed_requests'] == 1
        logs = client.get('/inference-logs').json()['logs']
        assert logs[0]['version'] is None and logs[0]['error_code'] == 'routing_unavailable'


def test_duplicate_active_lifespan_is_rejected_without_disrupting_owner(settings):
    app = create_app(settings)
    with TestClient(app) as client:
        with pytest.raises(errors.StartupRecoveryError,match='already active'):
            with TestClient(app):
                pytest.fail('duplicate lifespan started')
        assert client.get('/health').status_code == 200


def test_application_container_is_unavailable_until_recovery_completes(settings, monkeypatch):
    started, release = Event(), Event()
    original = ServiceContainer.reconcile_startup
    def paused(container):
        started.set()
        assert release.wait(5)
        return original(container)
    monkeypatch.setattr(ServiceContainer,'reconcile_startup',paused)
    app = create_app(settings)
    def start():
        with TestClient(app) as client:
            return client.get('/health').status_code
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(start)
        assert started.wait(5)
        try:
            assert app.state.container is None and app.state.startup_report is None
        finally:
            release.set()
        assert pending.result(timeout=10) == 200


def test_stale_failed_recovery_write_is_isolated_and_cached_runtime_detached(settings, sklearn_artifact, monkeypatch):
    prepare(settings,sklearn_artifact)
    with ServiceContainer(settings) as container:
        retained = container.runtime_manager.get_runtime('model','v1')
        container.artifact_storage.delete(container.model_registry.get_version('model','v1').artifact_location)
        def stop_then_fail(name,version):
            container.deployment_manager.undeploy('v1')
            raise errors.ArtifactNotFoundError('unavailable')
        monkeypatch.setattr(container.runtime_manager,'verify_available',stop_then_fail)
        app = create_app(container=container)
        with TestClient(app) as client:
            failure = app.state.startup_report.failures[0].error
            assert isinstance(failure,errors.DeploymentRecoveryError)
            assert isinstance(failure.persistence_error,errors.DeploymentConflictError)
            assert container.runtime_manager.cache_info().current_size == 0
            assert container.deployment_manager.get_deployment('v1').status is DeploymentStatus.STOPPED
            assert client.post('/predict/model',json={'version':'v1','inputs':[[4]]}).status_code == 503
            # Detaching the cache cannot corrupt a runtime already held by a caller.
            assert retained.predict([[4]]) == pytest.approx([9])
