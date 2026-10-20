"""Real security dependencies and bounded HTTP execution with isolated storage."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from secrets import token_urlsafe
from threading import Event

from fastapi.testclient import TestClient
import pytest

from app.api.dependencies import get_inference_service
from app.config import Settings
from app.domain.enums import ModelFramework
from app.domain.resource_config import ResourceConfig
from app.main import create_app
from app.services.inference_service import InferenceResult


@pytest.fixture
def settings(tmp_path):
    return Settings(database_url=f"sqlite:///{tmp_path / 'security.db'}", artifact_storage_dir=tmp_path/'artifacts',
                    admin_api_key=token_urlsafe(32), operator_api_key=token_urlsafe(32),
                    trust_sklearn_artifacts=True, max_concurrent_inference=1, max_waiting_inference=0)


@pytest.mark.parametrize('path', ['/metrics','/metrics/model','/metrics/model/versions','/inference-logs',
                                  '/runtime/resources','/runtime/admission','/routing/model',
                                  '/routing/model/history','/deployments/missing/health','/openapi.json','/docs'])
def test_operational_endpoints_fail_closed(settings, path):
    with TestClient(create_app(settings), client=('127.0.0.1',1234)) as client:
        response = client.get(path)
        assert response.status_code == 401
        assert response.json() == {'error':{'code':'http_error','message':'HTTP request could not be completed'}}


@pytest.mark.parametrize('key', ['', 'wrong', ' leading-space', 'x'*257])
def test_malformed_credentials(settings, key):
    with TestClient(create_app(settings), client=('127.0.0.1',1234)) as client:
        assert client.get('/metrics', headers={'X-API-Key':key}).status_code == 401


def test_duplicate_credentials_are_rejected(settings):
    with TestClient(create_app(settings), client=('127.0.0.1',1234)) as client:
        assert client.get('/metrics', headers=[('X-API-Key',settings.admin_api_key),('X-API-Key','wrong')]).status_code == 401


def test_unconfigured_credentials_fail_closed(settings):
    with TestClient(create_app(replace(settings,admin_api_key=None,operator_api_key=None)), client=('127.0.0.1',1234)) as client:
        assert client.get('/metrics',headers={'X-API-Key':'x'*32}).status_code == 401
        assert client.get('/health').status_code == 200


@pytest.mark.parametrize('host', ['203.0.113.5','testclient','::ffff:203.0.113.5'])
def test_remote_peer_rejected_even_with_key_and_forwarded_header(settings, host):
    with TestClient(create_app(settings),client=(host,1234),headers={'X-API-Key':settings.admin_api_key}) as client:
        assert client.get('/metrics',headers={'X-Forwarded-For':'127.0.0.1'}).status_code == 403


@pytest.mark.parametrize('host', ['127.0.0.1','::1','::ffff:127.0.0.1'])
@pytest.mark.parametrize('role', ['admin_api_key','operator_api_key'])
def test_read_only_access(settings, host, role):
    with TestClient(create_app(settings),client=(host,1234),headers={'X-API-Key':getattr(settings,role)}) as client:
        assert client.get('/metrics').status_code == 200
        assert client.get('/runtime/resources').status_code == 200
        assert client.get('/runtime/admission').json()['capacity'] == 1


@pytest.mark.parametrize('method,path,body', [('put','/routing/model',{'weights':{'v1':100},'expected_revision':0}),
                                           ('post','/routing/model/rollback',{'target_revision':1,'expected_revision':1})])
def test_operator_cannot_mutate_but_admin_reaches_service(settings, method, path, body):
    with TestClient(create_app(settings),client=('127.0.0.1',1234)) as client:
        assert getattr(client,method)(path,json=body,headers={'X-API-Key':settings.operator_api_key}).status_code == 403
        assert getattr(client,method)(path,json=body,headers={'X-API-Key':settings.admin_api_key}).status_code == 404


def test_openapi_and_docs_are_protected_and_describe_security(settings):
    with TestClient(create_app(settings),client=('127.0.0.1',1234),headers={'X-API-Key':settings.admin_api_key}) as client:
        spec = client.get('/openapi.json').json()
        assert spec['components']['securitySchemes']['APIKeyHeader']['name'] == 'X-API-Key'
        for path in ['/metrics','/runtime/admission','/routing/{model_name}']:
            assert spec['paths'][path]['get']['security'] == [{'APIKeyHeader':[]}]
        assert 'security' not in spec['paths']['/predict/{model_name}']['post']
        assert client.get('/docs').status_code == 200
        assert settings.admin_api_key not in client.get('/docs').text
        assert client.get('/redoc').status_code == 404


def test_overload_does_not_create_inference_observation_and_recovers(settings, sklearn_artifact):
    app = create_app(settings)
    with TestClient(app,client=('127.0.0.1',1234),headers={'X-API-Key':settings.admin_api_key}) as client:
        container = app.state.container
        container.model_registry.register_model('model')
        container.model_registry.register_version('model','v1',ModelFramework.SKLEARN,sklearn_artifact)
        container.deployment_manager.deploy('model','v1',ResourceConfig(1,128),deployment_id='d1')
        runtime = container.runtime_manager.get_runtime('model','v1')
        original = runtime.predict
        started, release = Event(), Event()
        def slow(inputs):
            started.set()
            assert release.wait(5)
            return original(inputs)
        runtime.predict = slow
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(client.post,'/predict/model',json={'version':'v1','inputs':[[4.0]]})
            assert started.wait(5)
            try:
                rejected = client.post('/predict/model/batch',json={'version':'v1','inputs':[[4.0],[5.0]]})
                assert rejected.status_code == 503
                assert rejected.json()['error']['code'] == 'inference_capacity_unavailable'
                assert 'Retry-After' not in rejected.headers
                assert client.get('/metrics/model').json()['total_requests'] == 0
                counters = client.get('/runtime/admission').json()
                assert counters['active'] == 1 and counters['capacity_rejections'] == 1
            finally:
                release.set()
            assert first.result(timeout=5).status_code == 200
        response = client.post('/predict/model',json={'version':'v1','inputs':[[5.0]]})
        assert response.status_code == 200 and response.json()['predictions'] == pytest.approx([11.0])
        assert client.get('/metrics/model').json()['total_requests'] == 2
        assert client.get('/runtime/admission').json()['active'] == 0


def test_public_prediction_and_sanitized_unexpected_errors(settings, caplog):
    class Fake:
        def predict(self,*args):
            raise RuntimeError('private artifact/path secret inputs')
    app = create_app(settings)
    app.dependency_overrides[get_inference_service] = lambda: Fake()
    with TestClient(app,client=('127.0.0.1',1234),raise_server_exceptions=True) as client:
        response = client.post('/predict/model',json={'version':'v1','inputs':[[4.0]]})
        assert response.status_code == 500
        assert 'private' not in response.text
        assert 'private' not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)
        assert sum('Unhandled API error' in record.message for record in caplog.records) == 1


@pytest.mark.parametrize('name,value', [('max_concurrent_inference',0),('max_concurrent_inference',True),
    ('max_waiting_inference',-1),('max_waiting_inference',False),('admission_wait_timeout_seconds',0),
    ('admission_wait_timeout_seconds',float('nan')),('admission_wait_timeout_seconds',float('inf')),
    ('admin_api_key','short'),('admin_api_key','x '*32),('operator_api_key','é'*32)])
def test_settings_reject_invalid_security_limits(settings, name, value):
    with pytest.raises(ValueError):
        replace(settings,**{name:value})


def test_security_environment_and_no_secrets_in_repr(monkeypatch):
    admin, operator = token_urlsafe(32), token_urlsafe(32)
    for name,value in {'ADMIN_API_KEY':admin,'OPERATOR_API_KEY':operator,'MAX_CONCURRENT_INFERENCE':'2',
                       'MAX_WAITING_INFERENCE':'0','ADMISSION_WAIT_TIMEOUT_SECONDS':'0.5'}.items():
        monkeypatch.setenv(name,value)
    settings = Settings()
    assert settings.admin_api_key == admin and settings.operator_api_key == operator
    assert settings.max_concurrent_inference == 2 and settings.max_waiting_inference == 0
    assert settings.admission_wait_timeout_seconds == .5
    assert admin not in repr(settings) and operator not in repr(settings)
    with pytest.raises(ValueError):
        replace(settings,operator_api_key=admin)


def test_cancelled_http_request_cannot_free_native_capacity(settings):
    import httpx
    async def scenario():
        started, release = Event(), Event()
        class Fake:
            def predict(self,*args):
                started.set()
                assert release.wait(5)
                return InferenceResult('model','v1',[9.0],1.0)
        app = create_app(settings)
        app.dependency_overrides[get_inference_service] = lambda: Fake()
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app,client=('127.0.0.1',1234))
            async with httpx.AsyncClient(transport=transport,base_url='http://test') as client:
                first = asyncio.create_task(client.post('/predict/model',json={'version':'v1','inputs':[[4.0]]}))
                assert await asyncio.to_thread(started.wait,5)
                try:
                    first.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await first
                    assert app.state.inference_admission.snapshot()['active'] == 1
                    assert (await client.post('/predict/model',json={'version':'v1','inputs':[[4.0]]})).status_code == 503
                finally:
                    release.set()
        assert app.state.container is None and app.state.inference_admission is None
    asyncio.run(scenario())


def test_waiting_request_rechecks_deployment_after_undeploy(settings, sklearn_artifact):
    settings = replace(settings,max_waiting_inference=1)
    app = create_app(settings)
    with TestClient(app,client=('127.0.0.1',1234),headers={'X-API-Key':settings.admin_api_key}) as client:
        container = app.state.container
        container.model_registry.register_model('model')
        container.model_registry.register_version('model','v1',ModelFramework.SKLEARN,sklearn_artifact)
        container.deployment_manager.deploy('model','v1',ResourceConfig(1,128),deployment_id='d1')
        runtime = container.runtime_manager.get_runtime('model','v1')
        original = runtime.predict
        started, release = Event(), Event()
        def slow(inputs):
            started.set()
            assert release.wait(5)
            return original(inputs)
        runtime.predict = slow
        async def queued():
            async def wait():
                while app.state.inference_admission.snapshot()['waiting'] != 1:
                    await asyncio.sleep(0)
            await asyncio.wait_for(wait(),5)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.post,'/predict/model',json={'version':'v1','inputs':[[4.0]]})
            assert started.wait(5)
            second = pool.submit(client.post,'/predict/model',json={'version':'v1','inputs':[[5.0]]})
            try:
                client.portal.call(queued)
                container.deployment_manager.undeploy('d1')
            finally:
                release.set()
            assert first.result(timeout=5).status_code == 200
            assert second.result(timeout=5).status_code == 503
        metrics = client.get('/metrics/model').json()
        assert metrics['total_requests'] == 2 and metrics['failed_requests'] == 1
        assert client.get('/runtime/admission').json()['active'] == 0


def test_http_admission_timeout_has_sanitized_503_and_preserves_active_work(settings):
    settings = replace(settings,max_waiting_inference=1,admission_wait_timeout_seconds=.001)
    started, release = Event(), Event()
    class Fake:
        def predict(self,*args):
            started.set()
            assert release.wait(5)
            return InferenceResult('model','v1',[9.0],1.0)
    app = create_app(settings)
    app.dependency_overrides[get_inference_service] = lambda: Fake()
    with TestClient(app,client=('127.0.0.1',1234),headers={'X-API-Key':settings.operator_api_key}) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(client.post,'/predict/model',json={'version':'v1','inputs':[[4.0]]})
            assert started.wait(5)
            try:
                expired = client.post('/predict/model',json={'version':'v1','inputs':[[4.0]]})
                assert expired.status_code == 503
                assert expired.json()['error']['code'] == 'inference_capacity_unavailable'
                counters = client.get('/runtime/admission').json()
                assert counters['active'] == 1 and counters['waiting'] == 0 and counters['wait_timeouts'] == 1
                assert counters['capacity_rejections'] == 0
            finally:
                release.set()
            assert first.result(timeout=5).status_code == 200
        assert client.get('/runtime/admission').json()['active'] == 0


def test_unexpected_response_failure_suppresses_private_exception_context(caplog):
    from app.api.middleware import SanitizedErrorMiddleware
    async def scenario():
        async def broken(scope, receive, send):
            await send({'type':'http.response.start','status':200,'headers':[]})
            raise RuntimeError('private artifact/path and payload')
        sent = []
        async def send(message):
            sent.append(message)
        async def receive():
            return {'type':'http.request','body':b''}
        with pytest.raises(RuntimeError,match='HTTP response could not be completed') as captured:
            await SanitizedErrorMiddleware(broken)({'type':'http'},receive,send)
        assert captured.value.__suppress_context__ is True
        assert captured.value.__cause__ is None
        assert len(sent) == 1
    asyncio.run(scenario())
    assert 'private' not in caplog.text
    assert len(caplog.records) == 1 and caplog.records[0].exc_info is None
