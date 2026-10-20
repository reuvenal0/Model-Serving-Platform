"""Routing persistence, orchestration, races, and real HTTP prediction."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import pickle
from random import Random
from threading import Barrier, Event
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest
from sklearn.linear_model import LinearRegression
from sqlalchemy import event, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.api.dependencies import get_routing_manager
from app.config import Settings
from app.core.container import ServiceContainer
from app.core import exceptions as errors
from app.domain.enums import DeploymentStatus, ModelFramework, ModelVersionStatus
from app.domain.resource_config import ResourceConfig
from app.domain.routing_config import RoutingConfig
from app.infrastructure.database.database import create_sqlite_engine, initialize_database
from app.infrastructure.database.sqlite_routing_repository import SQLiteRoutingRepository
from app.main import create_app
from app.services.traffic_router import TrafficRouter
from app.services.weighted_random_strategy import WeightedRandomStrategy


@pytest.fixture
def system(tmp_path):
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'routing.db'}",
                        artifact_storage_dir=tmp_path / 'artifacts', trust_sklearn_artifacts=True)
    with ServiceContainer(settings, routing_strategy=WeightedRandomStrategy(Random(42))) as container:
        container.model_registry.register_model('linear')
        for version, slope in [('v1', 2), ('v2', 3)]:
            estimator = LinearRegression().fit([[1.0], [2.0], [3.0]], [slope + 1, 2*slope + 1, 3*slope + 1])
            container.model_registry.register_version('linear', version, ModelFramework.SKLEARN,
                                                      pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL))
            container.deployment_manager.deploy('linear', version, ResourceConfig(1, 128), deployment_id=version)
        yield container


@pytest.fixture
def client(system):
    with TestClient(create_app(container=system), raise_server_exceptions=False) as client:
        yield client


def configure(system, weights=None, expected=0):
    return system.routing_manager.configure_routing('linear', weights or {'v1': 90, 'v2': 10}, expected)


def test_initial_duplicate_and_missing(system):
    repo = system.routing_repository
    with pytest.raises(errors.RoutingNotFoundError):
        repo.get('linear')
    with pytest.raises(errors.RoutingNotFoundError):
        repo.history('linear')
    initial = configure(system)
    assert initial.revision == 1 and dict(initial.config.weights) == {'v1': 90, 'v2': 10}
    assert initial.config.updated_at.utcoffset().total_seconds() == 0
    with pytest.raises(errors.RoutingConflictError):
        repo.create(initial.config)
    with pytest.raises(errors.RoutingHistoryNotFoundError):
        repo.get_revision('linear', 99)
    assert len(repo.history('linear')) == 1


def test_update_history_and_restore(system):
    configure(system, {'v1': 100, 'v2': 0})
    configure(system, {'v1': 90, 'v2': 10}, 1)
    configure(system, {'v1': 50, 'v2': 50}, 2)
    restored = system.routing_manager.rollback('linear', 1, 3)
    assert restored.revision == 4 and restored.restored_from == 1
    assert dict(restored.config.weights) == {'v1': 100, 'v2': 0}
    history = system.routing_manager.get_routing_history('linear')
    assert [item.revision for item in history] == [1, 2, 3, 4]
    assert [dict(item.config.weights)['v2'] for item in history] == [0, 10, 50, 0]
    history[0].config.update_weights({'v2': 100})
    assert dict(system.routing_repository.get_revision('linear', 1).config.weights) == {'v1': 100, 'v2': 0}


def test_repository_recreation_and_cross_instance_cas(system):
    configure(system)
    engine = create_sqlite_engine(system.settings.database_url)
    try:
        initialize_database(engine)
        other = SQLiteRoutingRepository(sessionmaker(engine))
        assert other.get('linear').revision == 1
        other.update(RoutingConfig('linear', {'v2': 100}), expected_revision=1)
        with pytest.raises(errors.RoutingConflictError):
            system.routing_repository.update(RoutingConfig('linear', {'v1': 100}), expected_revision=1)
        assert system.routing_repository.get('linear').revision == 2
        assert len(other.history('linear')) == 2
    finally:
        engine.dispose()


def test_concurrent_repository_writers_only_one_wins(system):
    configure(system)
    barrier = Barrier(2)
    engines = [create_sqlite_engine(system.settings.database_url) for _ in range(2)]

    def write(engine, version):
        repo = SQLiteRoutingRepository(sessionmaker(engine))
        expected = repo.get('linear').revision
        barrier.wait(timeout=5)
        try:
            return repo.update(RoutingConfig('linear', {version: 100}), expected_revision=expected).revision
        except errors.RoutingConflictError:
            return 'conflict'
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(write, engine, version) for engine, version in zip(engines, ['v1', 'v2'])]
            assert sorted([future.result(timeout=10) for future in futures], key=str) == [2, 'conflict']
        assert len(system.routing_repository.history('linear')) == 2
    finally:
        for engine in engines:
            engine.dispose()


def test_append_failure_rolls_back_current_and_history(system, monkeypatch):
    configure(system)
    original = system.routing_repository._append

    def fail(*args):
        original(*args)
        raise RuntimeError('injected failure after history insertion')
    monkeypatch.setattr(system.routing_repository, '_append', fail)
    with pytest.raises(RuntimeError):
        system.routing_repository.update(RoutingConfig('linear', {'v2': 100}), expected_revision=1)
    assert system.routing_repository.get('linear').revision == 1
    assert len(system.routing_repository.history('linear')) == 1


@pytest.mark.parametrize('table,operation', [('routing_history', 'UPDATE'), ('routing_history', 'DELETE'),
                                          ('routing_version_references', 'UPDATE'), ('routing_version_references', 'DELETE')])
def test_database_history_is_immutable(system, table, operation):
    configure(system)
    with pytest.raises(IntegrityError):
        with system.engine.begin() as connection:
            statement = f'UPDATE {table} SET revision=revision' if operation == 'UPDATE' else f'DELETE FROM {table}'
            connection.execute(text(statement))
    assert len(system.routing_repository.history('linear')) == 1


def test_invalid_persisted_weights_are_not_adopted(system):
    configure(system)
    with system.engine.begin() as connection:
        connection.execute(text("UPDATE routing_current SET weights = :weights"), {'weights': '{"v1": 3}'})
    with pytest.raises(errors.RepositoryError):
        system.routing_repository.get('linear')


@pytest.mark.parametrize('revision', [-1, True, 1.0, None])
def test_invalid_expected_revision(system, revision):
    with pytest.raises(errors.DomainValidationError):
        configure(system, expected=revision)


def test_unknown_models_versions_and_missing_configuration(system):
    with pytest.raises(errors.ModelNotFoundError):
        system.routing_manager.configure_routing('unknown', {'v1': 100}, 0)
    with pytest.raises(errors.ModelVersionNotFoundError):
        configure(system, {'unknown': 100})
    with pytest.raises(errors.ModelVersionNotFoundError):
        configure(system, {'v1': 100, 'unknown': 0})
    with pytest.raises(errors.RoutingNotFoundError):
        configure(system, expected=1)
    with pytest.raises(errors.RoutingNotFoundError):
        system.traffic_router.select('linear')


@pytest.mark.parametrize('weights', [{}, {'v1': 99}, {'v1': True}, {'v1': -1, 'v2': 101}, {'': 100}])
def test_manager_reuses_domain_validation(system, weights):
    with pytest.raises(errors.DomainValidationError):
        system.routing_manager.configure_routing('linear', weights, 0)


@pytest.mark.parametrize('status', [DeploymentStatus.STOPPED, DeploymentStatus.FAILED, DeploymentStatus.DEPLOYING])
def test_nonrunning_positive_target_rejected(system, status):
    snapshot = system.deployment_repository.get_deployment('v2')
    system.deployment_repository.update_deployment(replace(snapshot.deployment, status=status), expected_revision=snapshot.revision)
    with pytest.raises(errors.RoutingUnavailableError):
        configure(system)
    assert configure(system, {'v1': 100, 'v2': 0}).revision == 1


def test_readiness_failure_does_not_persist_configuration(system, monkeypatch):
    def failure(*args, **kwargs):
        raise errors.DeploymentHealthCheckError('private readiness detail')
    monkeypatch.setattr(system.deployment_manager, 'check_health', failure)
    with pytest.raises(errors.DeploymentHealthCheckError):
        configure(system)
    with pytest.raises(errors.RoutingNotFoundError):
        system.routing_repository.get('linear')


def test_stale_manager_and_rollback_are_rejected_before_health_checks(system, monkeypatch):
    configure(system)
    check = Mock()
    monkeypatch.setattr(system.deployment_manager, 'check_health', check)
    with pytest.raises(errors.RoutingConflictError):
        configure(system, expected=0)
    with pytest.raises(errors.RoutingConflictError):
        system.routing_manager.rollback('linear', 1, 0)
    check.assert_not_called()


def test_rollback_unavailable_target_is_rejected(system):
    configure(system, {'v2': 100})
    configure(system, {'v1': 100}, 1)
    system.deployment_manager.undeploy('v2')
    with pytest.raises(errors.RoutingUnavailableError):
        system.routing_manager.rollback('linear', 1, 2)
    assert len(system.routing_repository.history('linear')) == 2
    assert dict(system.routing_repository.get('linear').config.weights) == {'v1': 100}


def test_stop_during_configuration_readiness_prevents_write(system, monkeypatch):
    original = system.deployment_manager.check_health

    def stop_after_check(identifier):
        result = original(identifier)
        system.deployment_manager.undeploy(identifier)
        return result
    monkeypatch.setattr(system.deployment_manager, 'check_health', stop_after_check)
    with pytest.raises(errors.RoutingUnavailableError):
        configure(system)
    with pytest.raises(errors.RoutingNotFoundError):
        system.routing_repository.get('linear')


def test_stop_after_validation_can_commit_but_requests_reject(system, monkeypatch):
    original = system.routing_repository.create

    def race(config):
        system.deployment_manager.undeploy('v2')
        return original(config)
    monkeypatch.setattr(system.routing_repository, 'create', race)
    assert configure(system).revision == 1
    with pytest.raises(errors.RoutingUnavailableError):
        system.inference_service.predict('linear', None, [[4.0]])


def test_strict_policy_never_renormalizes_or_invokes_strategy(system, monkeypatch):
    configure(system)
    system.deployment_manager.undeploy('v1')
    strategy = Mock()
    monkeypatch.setattr(system.traffic_router, '_strategy', strategy)
    with pytest.raises(errors.RoutingUnavailableError):
        system.traffic_router.select('linear')
    strategy.select.assert_not_called()
    assert system.inference_service.predict('linear', 'v2', [[4.0]]).predictions == pytest.approx([13.0])


def test_router_never_loads_models(system, monkeypatch):
    configure(system)
    monkeypatch.setattr(system.runtime_manager, 'get_runtime', Mock(side_effect=AssertionError('router loaded a model')))
    assert system.traffic_router.select('linear').version in ('v1', 'v2')


def test_explicit_bypasses_router_and_automatic_batch_selects_once(system, monkeypatch):
    configure(system, {'v2': 100})
    select = Mock(wraps=system.traffic_router.select)
    monkeypatch.setattr(system.traffic_router, 'select', select)
    first = system.runtime_manager.get_runtime('linear', 'v2')
    assert system.inference_service.predict('linear', 'v1', [[4.0]]).predictions == pytest.approx([9.0])
    select.assert_not_called()
    predict = Mock(wraps=first.predict)
    monkeypatch.setattr(first, 'predict', predict)
    result = system.inference_service.batch_predict('linear', None, [[4.0], [5.0]])
    assert result.version == 'v2' and result.predictions == pytest.approx([13.0, 16.0])
    select.assert_called_once_with('linear')
    predict.assert_called_once_with([[4.0], [5.0]])
    assert system.runtime_manager.get_runtime('linear', 'v2') is first
    assert system.runtime_manager.cache_info().loads == 2


def test_selected_destination_stopped_before_admission_no_fallback(system, monkeypatch):
    configure(system, {'v1': 50, 'v2': 50})
    original = system.traffic_router.select

    def stop(model_name):
        choice = original(model_name)
        system.deployment_manager.undeploy(choice.version)
        return choice
    monkeypatch.setattr(system.traffic_router, 'select', stop)
    with pytest.raises(errors.InferenceUnavailableError):
        system.inference_service.predict('linear', None, [[4.0]])


def test_actual_concurrent_stop_between_selection_and_admission(system, monkeypatch):
    configure(system, {'v1': 100})
    entered, release = Event(), Event()
    original = system.traffic_router.select

    def blocked(name):
        choice = original(name)
        entered.set()
        assert release.wait(5)
        return choice
    monkeypatch.setattr(system.traffic_router, 'select', blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(system.inference_service.predict, 'linear', None, [[4.0]])
        assert entered.wait(5)
        try:
            system.deployment_manager.undeploy('v1')
        finally:
            release.set()
        with pytest.raises(errors.InferenceUnavailableError):
            pending.result(timeout=10)


def test_history_references_block_version_deletion_even_zero_weight(system):
    registry = system.model_registry
    artifact = registry.download_artifact('linear', 'v1')
    registry.register_version('linear', 'inactive', ModelFramework.SKLEARN, artifact)
    configure(system, {'v1': 100, 'inactive': 0})
    configure(system, {'v1': 100}, 1)
    with pytest.raises(errors.ModelVersionInUseError):
        registry.delete_version('linear', 'inactive')
    assert registry.download_artifact('linear', 'inactive') == artifact


def test_repository_unknown_reference_rolls_back_current_update(system):
    configure(system)
    with pytest.raises(errors.ModelVersionNotFoundError):
        system.routing_repository.update(RoutingConfig('linear', {'missing': 100}), expected_revision=1)
    assert system.routing_repository.get('linear').revision == 1
    assert len(system.routing_repository.history('linear')) == 1


def test_management_http_and_automatic_prediction(client):
    assert client.get('/routing/linear').status_code == 404
    assert client.post('/predict/linear', json={'inputs': [[4.0]]}).status_code == 404
    first = client.put('/routing/linear', json={'weights': {'v1': 90, 'v2': 10}, 'expected_revision': 0})
    assert first.status_code == 200 and first.json()['revision'] == 1
    assert client.get('/routing/linear').json() == first.json()
    second = client.put('/routing/linear', json={'weights': {'v2': 100}, 'expected_revision': 1})
    assert second.status_code == 200 and second.json()['revision'] == 2
    for endpoint in ['/predict/linear', '/predict/linear/batch']:
        body = client.post(endpoint, json={'inputs': [[4.0], [5.0]]}).json()
        assert body['version'] == 'v2' and body['predictions'] == pytest.approx([13, 16])
    assert client.post('/predict/linear', json={'version': None, 'inputs': [[4.0]]}).json()['version'] == 'v2'
    rollback = client.post('/routing/linear/rollback', json={'target_revision': 1, 'expected_revision': 2})
    assert rollback.status_code == 200 and rollback.json()['revision'] == 3
    assert rollback.json()['restored_from'] == 1
    assert [item['revision'] for item in client.get('/routing/linear/history').json()] == [1, 2, 3]
    assert client.put('/routing/linear', json={'weights': {'v1': 100}, 'expected_revision': 1}).status_code == 409
    assert client.post('/routing/linear/rollback', json={'target_revision': 99, 'expected_revision': 3}).status_code == 404


@pytest.mark.parametrize('body', [
    {}, {'weights': {'v1': 100}}, {'weights': {'v1': 100}, 'expected_revision': True},
    {'weights': {'v1': 99}, 'expected_revision': 0}, {'weights': {'v1': True}, 'expected_revision': 0},
    {'weights': {'v1': 100.0}, 'expected_revision': 0}, {'weights': {'v1': 100}, 'expected_revision': -1},
    {'weights': {'v1': 100}, 'expected_revision': 0, 'extra': True},
    {'weights': [['v1', 100]], 'expected_revision': 0},
])
def test_invalid_routing_http_requests(client, body):
    assert client.put('/routing/linear', json=body).status_code == 422


def test_duplicate_json_weights_rejected(client):
    body = '{"weights":{"v1":90,"v1":100},"expected_revision":0}'
    response = client.put('/routing/linear', content=body, headers={'content-type': 'application/json'})
    assert response.status_code == 422
    assert client.get('/routing/linear').status_code == 404


def test_http_unknown_and_unavailable_targets(client, system):
    assert client.put('/routing/missing', json={'weights': {'v1': 100}, 'expected_revision': 0}).status_code == 404
    assert client.put('/routing/linear', json={'weights': {'missing': 100}, 'expected_revision': 0}).status_code == 404
    configure(system)
    system.deployment_manager.undeploy('v2')
    assert client.post('/predict/linear', json={'inputs': [[4]]}).status_code == 503
    assert client.post('/predict/linear', json={'version': 'v1', 'inputs': [[4]]}).status_code == 200
    assert client.put('/routing/linear', json={'weights': {'v2': 100}, 'expected_revision': 1}).status_code == 503


@pytest.mark.parametrize('error,status', [(errors.RoutingConflictError('/private/db'), 409),
                                        (errors.RoutingHistoryNotFoundError('/private/revision'), 404),
                                        (errors.RoutingUnavailableError('/private/model'), 503)])
def test_sanitized_routing_errors_and_dependency_override(client, error, status):
    fake = Mock()
    fake.get_routing_config.side_effect = error
    client.app.dependency_overrides[get_routing_manager] = lambda: fake
    response = client.get('/routing/linear')
    assert response.status_code == status and 'private' not in response.text
    assert set(response.json()['error']) == {'code', 'message'}


def test_migration_preserves_legacy_data_and_is_idempotent(tmp_path):
    from app.infrastructure.database.records import Base, DeploymentRecord, ModelRecord, ModelVersionRecord
    from app.infrastructure.database.sqlite_model_repository import SQLiteModelRepository
    from app.domain.model import Model

    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    try:
        # Simulate a Phase 5 database without routing tables or a migration ledger.
        Base.metadata.create_all(engine, tables=[ModelRecord.__table__, ModelVersionRecord.__table__, DeploymentRecord.__table__])
        repository = SQLiteModelRepository(sessionmaker(engine))
        repository.save_model(Model('preserved', 'legacy metadata'))
        before = repository.get_model('preserved')
        initialize_database(engine)
        initialize_database(engine)
        assert repository.get_model('preserved') == before
        with engine.connect() as connection:
            assert connection.scalar(text('SELECT COUNT(*) FROM schema_migrations WHERE version=6')) == 1
    finally:
        engine.dispose()


def test_repository_failure_at_reference_insert_is_atomic(system):
    configure(system)

    def fail_reference_insert(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith('INSERT INTO routing_version_references'):
            raise RuntimeError('injected reference write failure')
    event.listen(system.engine, 'before_cursor_execute', fail_reference_insert)
    try:
        with pytest.raises(RuntimeError):
            system.routing_repository.update(RoutingConfig('linear', {'v2': 100}), expected_revision=1)
    finally:
        event.remove(system.engine, 'before_cursor_execute', fail_reference_insert)
    assert system.routing_repository.get('linear').revision == 1
    assert len(system.routing_repository.history('linear')) == 1


def test_actual_readiness_failure_marks_failed_and_blocks_config(system):
    artifact = system.model_registry.get_version('linear', 'v2').artifact_location
    system.artifact_storage.delete(artifact)
    system.runtime_manager.clear()
    with pytest.raises(errors.ArtifactNotFoundError):
        configure(system)
    assert system.deployment_manager.get_deployment('v2').status is DeploymentStatus.FAILED
    with pytest.raises(errors.RoutingNotFoundError):
        system.routing_repository.get('linear')


def test_invalid_version_allowed_only_at_zero_weight(system):
    with system.engine.begin() as connection:
        connection.execute(text("UPDATE model_versions SET status='invalid' WHERE version='v2'"))
    with pytest.raises(errors.RoutingUnavailableError):
        configure(system)
    configure(system, {'v1': 100, 'v2': 0})
    assert system.traffic_router.select('linear').version == 'v1'


def test_routing_cas_rejects_writer_that_changes_during_health_checks(system, monkeypatch):
    configure(system)
    original = system.deployment_manager.check_health
    changed = False

    def competing_writer(identifier):
        nonlocal changed
        result = original(identifier)
        if not changed:
            changed = True
            system.routing_repository.update(RoutingConfig('linear', {'v1': 100}), expected_revision=1)
        return result
    monkeypatch.setattr(system.deployment_manager, 'check_health', competing_writer)
    with pytest.raises(errors.RoutingConflictError):
        configure(system, {'v2': 100}, 1)
    assert system.routing_repository.get('linear').revision == 2
    assert len(system.routing_repository.history('linear')) == 2


def test_health_readiness_does_not_add_prediction_calls(system, monkeypatch):
    runtimes = [system.runtime_manager.get_runtime('linear', version) for version in ('v1', 'v2')]
    predictions = [Mock(wraps=runtime.predict) for runtime in runtimes]
    for runtime, prediction in zip(runtimes, predictions):
        monkeypatch.setattr(runtime, 'predict', prediction)
    configure(system)
    system.inference_service.predict('linear', None, [[4]])
    assert sum(prediction.call_count for prediction in predictions) == 1
