"""Automatic restart recovery and resource safety with disposable local data."""

from pathlib import Path
from secrets import token_urlsafe
import pickle
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from app.config import Settings
from app.core.exceptions import ArtifactTooLargeError
from app.domain.enums import ModelFramework
from app.domain.resource_config import ResourceConfig
from app.main import create_app


def main() -> None:
    with TemporaryDirectory(prefix='recovery-demo-') as directory:
        root = Path(directory)
        settings = Settings(admin_api_key=token_urlsafe(32), database_url=f"sqlite:///{root / 'metadata.db'}",
                            artifact_storage_dir=root / 'artifacts',trust_sklearn_artifacts=True,
                            max_runtime_artifact_bytes=1024,runtime_cache_max_size=1)
        artifact = pickle.dumps(LinearRegression().fit([[1.0],[2.0],[3.0]],[3.0,5.0,7.0]),
                                protocol=pickle.HIGHEST_PROTOCOL)
        first = create_app(settings)
        with TestClient(first, client=("127.0.0.1", 50000), headers={"X-API-Key": settings.admin_api_key}):
            container = first.state.container
            container.model_registry.register_model('recovered-linear')
            container.model_registry.register_version('recovered-linear','v1',ModelFramework.SKLEARN,artifact)
            deployment = container.deployment_manager.deploy('recovered-linear','v1',ResourceConfig(1,128),deployment_id='demo')
            print('Before shutdown:',deployment.id,deployment.status.value)
        if not container._closed or first.state.container is not None:
            raise RuntimeError('First application did not close owned resources')
        print('First application shut down; metadata and artifact persist in the temporary environment.')
        second = create_app(settings)
        with TestClient(second, client=("127.0.0.1", 50000), headers={"X-API-Key": settings.admin_api_key}) as client:
            report = second.state.startup_report
            if report.failures or not all(item.ready for item in report.health):
                raise RuntimeError('Automatic startup recovery did not complete')
            print('Automatic startup reconciliation:',len(report.health),'ready,',len(report.failures),'failures')
            health = client.get('/deployments/demo/health')
            health.raise_for_status()
            if health.json()['ready'] is not True:
                raise RuntimeError('Recovered deployment is not ready')
            print('Readiness after restart:',health.json())
            response = client.post('/predict/recovered-linear',json={'version':'v1','inputs':[[4.0]]})
            response.raise_for_status()
            if abs(response.json()['predictions'][0]-9.0) > 1e-8:
                raise RuntimeError('Recovered prediction is incorrect')
            print('Prediction after restart:',response.json())
            resources = client.get('/runtime/resources')
            resources.raise_for_status()
            print('Process RSS and cache statistics:',resources.json())
            container = second.state.container
            container.model_registry.register_model('oversized-test')
            content = b'x'*(settings.max_runtime_artifact_bytes+1)
            container.model_registry.register_version('oversized-test','v1',ModelFramework.SKLEARN,content)
            try:
                container.runtime_manager.get_runtime('oversized-test','v1')
            except ArtifactTooLargeError:
                print('Oversized artifact rejected before deserialization: PASS')
            else:
                raise RuntimeError('Oversized artifact was accepted')
            if container.model_registry.download_artifact('oversized-test','v1') != content:
                raise RuntimeError('Oversized registered artifact was changed')
            print('Oversized artifact and registration metadata retained unchanged.')
        if not container._closed:
            raise RuntimeError('Second application did not close owned resources')
    print('Recovery demo passed; temporary database and artifacts removed.')


if __name__ == '__main__':
    main()
