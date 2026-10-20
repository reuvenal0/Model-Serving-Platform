"""End-to-end observations preserve routing, admission, and API responses."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import logging
import pickle
from random import Random
from unittest.mock import Mock, patch

from tests.api_client import TestClient
import pytest
from sklearn.linear_model import LinearRegression

from app.api.dependencies import get_metrics_collector
from app.config import Settings
from app.core.container import ServiceContainer
from app.core import exceptions as errors
from app.domain.enums import ModelFramework
from app.domain.resource_config import ResourceConfig
from app.infrastructure.database.sqlite_inference_log_repository import SQLiteInferenceLogRepository
from app.main import create_app
from app.services.metrics_collector import MetricsCollector
from app.services.weighted_random_strategy import WeightedRandomStrategy


@pytest.fixture
def settings(tmp_path):
    return Settings(database_url=f"sqlite:///{tmp_path / 'observations.db'}",
                    artifact_storage_dir=tmp_path / 'artifacts', trust_sklearn_artifacts=True)


@pytest.fixture
def system(settings):
    with ServiceContainer(settings, routing_strategy=WeightedRandomStrategy(Random(1))) as container:
        container.model_registry.register_model('linear')
        for version, slope in [('v1', 2), ('v2', 3)]:
            estimator = LinearRegression().fit([[1.0], [2.0], [3.0]], [slope+1,2*slope+1,3*slope+1])
            container.model_registry.register_version('linear', version, ModelFramework.SKLEARN,
                                                      pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL))
            container.deployment_manager.deploy('linear', version, ResourceConfig(1,128), deployment_id=version)
        container.routing_manager.configure_routing('linear', {'v1': 50, 'v2': 50}, 0)
        yield container


@pytest.fixture
def client(system):
    with TestClient(create_app(container=system), raise_server_exceptions=False) as client:
        yield client


def observations(system):
    return system.metrics_collector.query_logs().logs


def test_explicit_auto_and_batch_record_once_with_runtime_reuse(system):
    service = system.inference_service
    first = service.predict('linear', 'v1', [[4.0]])
    auto = service.predict('linear', None, [[4.0]])
    batch = service.batch_predict('linear', 'v2', [[4.0],[5.0]])
    logs = observations(system)
    assert len(logs) == 3 and len({log.request_id for log in logs}) == 3
    assert all(log.success and log.error_message is None and log.input_hash is None for log in logs)
    assert [log.version for log in logs] == ['v1',auto.version,'v2']
    assert logs[0].latency_ms == first.latency_ms
    assert batch.predictions == pytest.approx([13,16])
    assert system.runtime_manager.cache_info().loads == 2


@pytest.mark.parametrize('model,version,inputs,code,selected', [
    ('linear','v1',[], 'invalid_input','v1'),
    ('linear',None,[], 'invalid_input',None),
    ('linear','v1',[[1,2]], 'invalid_input','v1'),
    ('missing','v1',[[4]], 'not_found','v1'),
    ('linear','missing',[[4]], 'not_found','missing'),
    ('linear',' ',[[4]], 'invalid_input',None),
    (' ',None,[[4]], 'invalid_input',None),
])
def test_service_validation_and_lookup_failures_logged(system, model, version, inputs, code, selected):
    with pytest.raises(errors.DomainError):
        system.inference_service.predict(model, version, inputs)
    log, = observations(system)
    assert not log.success and log.error_message == code and log.version == selected
    if not model.strip():
        assert log.model_name is None


def test_routing_failure_before_selection_is_null(system):
    system.model_registry.register_model('no-routing')
    with pytest.raises(errors.RoutingNotFoundError):
        system.inference_service.predict('no-routing', None, [[4]])
    log, = observations(system)
    assert log.version is None and log.error_message == 'routing_not_found'


def test_deployment_and_routing_unavailability_logged_distinctly(system):
    system.deployment_manager.undeploy('v1')
    with pytest.raises(errors.InferenceUnavailableError):
        system.inference_service.predict('linear','v1',[[4]])
    with pytest.raises(errors.RoutingUnavailableError):
        system.inference_service.predict('linear',None,[[4]])
    logs = observations(system)
    assert [(log.version,log.error_message) for log in logs] == [('v1','unavailable'),(None,'routing_unavailable')]


@pytest.mark.parametrize('error,code', [(errors.ModelLoadingError('/private/model.pkl'),'runtime_loading_failed'),
                                      (errors.RepositoryError('/private/database'),'persistence_unavailable')])
def test_runtime_lookup_failures_preserved(system, monkeypatch, error, code):
    monkeypatch.setattr(system.runtime_manager, 'get_runtime', Mock(side_effect=error))
    with pytest.raises(type(error)) as caught:
        system.inference_service.predict('linear','v1',[[4]])
    assert caught.value is error
    log, = observations(system)
    assert log.error_message == code and '/private' not in repr(log)


@pytest.mark.parametrize('error,code', [(errors.PredictionError('private payload'),'prediction_failed'),
                                      (errors.PredictionInputError('private payload'),'invalid_input'),
                                      (RuntimeError('private payload'),'prediction_failed')])
def test_prediction_failures_classified_without_content(system, monkeypatch, error, code):
    runtime = system.runtime_manager.get_runtime('linear','v1')
    monkeypatch.setattr(runtime,'predict',Mock(side_effect=error))
    with pytest.raises(errors.PredictionError):
        system.inference_service.predict('linear','v1',[[4]])
    log, = observations(system)
    assert log.error_message == code and 'private payload' not in repr(log)


def test_output_normalization_failure_is_observed(system, monkeypatch):
    runtime = system.runtime_manager.get_runtime('linear','v1')
    monkeypatch.setattr(runtime,'predict',lambda inputs: object())
    with pytest.raises(errors.PredictionSerializationError):
        system.inference_service.predict('linear','v1',[[4]])
    assert observations(system)[0].error_message == 'serialization_failed'


def test_best_effort_write_failure_preserves_success_and_original_error(system, monkeypatch, caplog):
    failure = errors.RepositoryError('/private/password')
    write = Mock(side_effect=failure)
    monkeypatch.setattr(system.inference_log_repository,'append_log',write)
    with caplog.at_level(logging.ERROR, logger='app.services.inference_service'):
        assert system.inference_service.predict('linear','v1',[[4]]).predictions == pytest.approx([9])
        original = errors.PredictionError('original private error')
        monkeypatch.setattr(system.runtime_manager.get_runtime('linear','v1'),'predict',Mock(side_effect=original))
        with pytest.raises(errors.PredictionError) as caught:
            system.inference_service.predict('linear','v1',[[4]])
    assert caught.value is original and write.call_count == 2
    diagnostics = [record for record in caplog.records if record.name == 'app.services.inference_service']
    assert len(diagnostics) == 2  # One diagnostic for each failed append, no duplicates.
    for record in diagnostics:
        assert record.levelno == logging.ERROR
        assert record.getMessage() == 'monitoring_write_failed: observation persistence was not confirmed'
        assert record.args == () and record.exc_info is None and record.stack_info is None
    assert not observations(system)


def test_unique_observations_under_concurrent_cached_inference(system):
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: system.inference_service.predict('linear','v1',[[4]]), range(32)))
    logs = observations(system)
    assert len(logs) == len({log.request_id for log in logs}) == 32
    assert system.runtime_manager.cache_info().loads == 2


def test_monotonic_latency_excludes_monitoring_write(system, monkeypatch):
    monkeypatch.setattr('app.services.inference_service.perf_counter_ns', Mock(side_effect=[100, 5000100]))
    result = system.inference_service.predict('linear','v1',[[4]])
    assert result.latency_ms == observations(system)[0].latency_ms == 5


def test_duplicate_observation_id_never_changes_prediction(system, caplog):
    with patch('app.services.inference_service.uuid4') as uuid:
        uuid.return_value.hex = 'fixed-id'
        system.inference_service.predict('linear','v1',[[4]])
        system.inference_service.predict('linear','v1',[[5]])
    assert len(observations(system)) == 1
    diagnostics = [record for record in caplog.records if record.name == 'app.services.inference_service']
    assert len(diagnostics) == 1
    assert diagnostics[0].levelno == logging.ERROR
    assert diagnostics[0].getMessage() == 'monitoring_write_failed: observation persistence was not confirmed'
    assert diagnostics[0].exc_info is None and diagnostics[0].stack_info is None


def test_health_http_validation_and_monitoring_reads_do_not_count(client, system):
    client.get('/health')
    client.get('/deployments/v1/health')
    assert client.post('/predict/linear', json={'version': ''}).status_code == 422
    assert client.get('/metrics').status_code == 200
    assert not observations(system)


def test_monitoring_endpoints_empty_filters_and_pagination(client):
    empty = client.get('/metrics').json()
    assert empty['total_requests'] == 0 and empty['error_rate'] == 0
    assert empty['average_latency_ms'] is None and empty['p99_latency_ms'] is None
    assert client.get('/metrics/linear/versions').json() == []
    for version in ('v1','v2'):
        assert client.post('/predict/linear',json={'version':version,'inputs':[[4]]}).status_code == 200
    assert client.post('/predict/linear',json={'version':'v1','inputs':[]}).status_code == 422
    assert client.get('/metrics/linear').json()['total_requests'] == 3
    assert client.get('/metrics/linear',params={'version':'v2'}).json()['total_requests'] == 1
    assert len(client.get('/metrics/linear/versions').json()) == 2
    assert client.get('/metrics/missing').json()['total_requests'] == 0
    first = client.get('/inference-logs',params={'limit':1}).json()
    assert first['next_cursor'] is not None and len(first['logs']) == 1
    second = client.get('/inference-logs',params={'limit':1,'cursor':first['next_cursor'],
                                                'start':first['start'],'end':first['end']}).json()
    assert first['logs'][0]['request_id'] != second['logs'][0]['request_id']
    failed = client.get('/inference-logs',params={'success':'false'}).json()['logs']
    assert len(failed) == 1 and failed[0]['error_code'] == 'invalid_input'
    assert not {'inputs','predictions','error_message','input_hash'}.intersection(failed[0])
    assert failed[0]['timestamp'].endswith(('Z','+00:00'))


@pytest.mark.parametrize('params', [{'limit':0},{'limit':201},{'cursor':-1}, {'start':'not-a-date'},
                                    {'start':'2026-01-01T00:00:00'},
                                    {'start':'2026-01-02T00:00:00Z','end':'2026-01-01T00:00:00Z'},
                                    {'unknown_version':True,'version':'v1'},{'version':' '}])
def test_monitoring_query_validation(client, params):
    response = client.get('/inference-logs',params=params)
    assert response.status_code == 422 and 'Traceback' not in response.text


def test_monitoring_backend_errors_are_sanitized_and_di_replaceable(client):
    fake = Mock(spec=MetricsCollector)
    fake.get_metrics.side_effect = errors.RepositoryError('/private/artifacts/model.pkl')
    client.app.dependency_overrides[get_metrics_collector] = lambda: fake
    response = client.get('/metrics')
    assert response.status_code == 503 and 'private' not in response.text


def test_live_api_optional_monitoring_checks_with_real_asgi_service(client):
    # The reusable live verifier also works with a real app through TestClient's
    # HTTP interface; its default address and ordinary httpx use remain unchanged.
    from examples.test_live_api import verify

    def redirect(method, path, **kwargs):
        path = path.replace('demo-linear','linear').replace('demo-api-deployment','v1')
        response = client.request(method,path,**kwargs)
        body = response.json()
        if isinstance(body,dict):
            if body.get('model_name') == 'linear':
                body['model_name'] = 'demo-linear'
            if body.get('deployment_id') == 'v1':
                body['deployment_id'] = 'demo-api-deployment'
        import httpx
        return httpx.Response(response.status_code,json=body)

    fake = Mock()
    fake.__enter__ = Mock(return_value=fake)
    fake.__exit__ = Mock(return_value=None)
    fake.request.side_effect = redirect
    with patch('examples.test_live_api.httpx.Client',return_value=fake):
        assert verify('http://127.0.0.1:8000','demo-api-deployment',False,True)


@pytest.mark.parametrize('start,end', [
    ('0001-01-01T00:00:00+01:00', '0001-01-01T00:05:00+01:00'),
    ('9999-12-31T23:54:00-01:00', '9999-12-31T23:59:00-01:00'),
])
def test_utc_normalization_overflow_returns_422(client, start, end):
    response = client.get('/metrics', params={'start': start, 'end': end})
    assert response.status_code == 422
    assert set(response.json()['error']) == {'code', 'message'}
    assert 'OverflowError' not in response.text and 'Traceback' not in response.text
