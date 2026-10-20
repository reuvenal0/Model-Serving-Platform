"""Isolated API-key and bounded native execution demonstration."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
from secrets import token_urlsafe
from tempfile import TemporaryDirectory
from threading import Event

from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from app.config import Settings
from app.domain.enums import ModelFramework
from app.domain.resource_config import ResourceConfig
from app.main import create_app


def main() -> None:
    with TemporaryDirectory(prefix='security-load-demo-') as directory:
        root = Path(directory)
        key = token_urlsafe(32)
        settings = Settings(database_url=f"sqlite:///{root / 'metadata.db'}", artifact_storage_dir=root/'artifacts',
                            trust_sklearn_artifacts=True, admin_api_key=key,
                            max_concurrent_inference=1,max_waiting_inference=0)
        app = create_app(settings)
        with TestClient(app,client=('127.0.0.1',50000)) as client:
            unauthorized = client.get('/runtime/admission')
            assert unauthorized.status_code == 401
            print('Unauthorized operational access:', unauthorized.status_code)
            client.headers['X-API-Key'] = key
            authorized = client.get('/runtime/admission')
            assert authorized.status_code == 200
            print('Authorized operational access:', authorized.status_code)
            container = app.state.container
            estimator = LinearRegression().fit([[1.0],[2.0],[3.0]],[3.0,5.0,7.0])
            container.model_registry.register_model('load-demo')
            container.model_registry.register_version('load-demo','v1',ModelFramework.SKLEARN,pickle.dumps(estimator))
            container.deployment_manager.deploy('load-demo','v1',ResourceConfig(1,128),deployment_id='demo')
            payload = {'version':'v1','inputs':[[4.0]]}
            normal = client.post('/predict/load-demo',json=payload)
            normal.raise_for_status()
            assert abs(normal.json()['predictions'][0]-9) < 1e-8
            print('Normal prediction:', normal.json())
            runtime = container.runtime_manager.get_runtime('load-demo','v1')
            original = runtime.predict
            started, release = Event(), Event()
            def controlled(inputs):
                started.set()
                if not release.wait(10):
                    raise RuntimeError('Demonstration synchronization expired')
                return original(inputs)
            runtime.predict = controlled
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(client.post,'/predict/load-demo',json=payload)
                assert started.wait(10)
                try:
                    rejected = client.post('/predict/load-demo',json=payload)
                    assert rejected.status_code == 503
                    counters = client.get('/runtime/admission').json()
                    assert counters['active'] == 1 and counters['capacity_rejections'] == 1
                    print('Controlled excess-load rejection:', rejected.status_code, rejected.json())
                    print('During load:', counters)
                finally:
                    release.set()
                pending.result(timeout=10).raise_for_status()
            client.post('/predict/load-demo',json=payload).raise_for_status()
            final = client.get('/runtime/admission').json()
            assert final['active'] == final['waiting'] == 0 and final['admitted_requests'] == 3
            assert client.get('/metrics/load-demo').json()['total_requests'] == 3
            print('Recovered admission counters:', final)
            print('PASS: three observations; overload was not classified as inference failure.')
        print('Shutdown drained workers; temporary resources cleaned up.')


if __name__ == '__main__':
    main()
