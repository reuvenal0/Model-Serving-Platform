"""Temporary, reproducible inference observations and persisted metrics."""

from dataclasses import asdict
from pathlib import Path
from secrets import token_urlsafe
import pickle
from random import Random
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.core.container import ServiceContainer
from app.domain.enums import ModelFramework
from app.domain.resource_config import ResourceConfig
from app.infrastructure.database.sqlite_inference_log_repository import SQLiteInferenceLogRepository
from app.main import create_app
from app.services.metrics_collector import MetricsCollector
from app.services.weighted_random_strategy import WeightedRandomStrategy


def main() -> None:
    with TemporaryDirectory(prefix='monitoring-demo-') as directory:
        root = Path(directory)
        settings = Settings(admin_api_key=token_urlsafe(32), database_url=f"sqlite:///{root / 'metadata.db'}",
                            artifact_storage_dir=root / 'artifacts', trust_sklearn_artifacts=True)
        with ServiceContainer(settings, routing_strategy=WeightedRandomStrategy(Random(42))) as container:
            registry = container.model_registry
            registry.register_model('observed-linear')
            for version, slope in [('v1',2),('v2',3)]:
                estimator = LinearRegression().fit([[1.0],[2.0],[3.0]], [slope+1,2*slope+1,3*slope+1])
                registry.register_version('observed-linear',version,ModelFramework.SKLEARN,
                                          pickle.dumps(estimator,protocol=pickle.HIGHEST_PROTOCOL))
                container.deployment_manager.deploy('observed-linear',version,ResourceConfig(1,128),deployment_id=version)
            container.routing_manager.configure_routing('observed-linear',{'v1':90,'v2':10},0)
            with TestClient(create_app(container=container), client=("127.0.0.1", 50000), headers={"X-API-Key": settings.admin_api_key}) as client:
                for version in ('v1','v2'):
                    response = client.post('/predict/observed-linear',json={'version':version,'inputs':[[4.0]]})
                    response.raise_for_status()
                    print('Explicit prediction:',response.json())
                batch = client.post('/predict/observed-linear/batch',json={'inputs':[[4.0],[5.0]]})
                batch.raise_for_status()
                print('Routed batch (one request):',batch.json())
                failure = client.post('/predict/observed-linear',json={'version':'v1','inputs':[]})
                if failure.status_code != 422:
                    raise RuntimeError('Expected controlled inference validation failure')
                print('Controlled failure:',failure.status_code,failure.json())
                logs = client.get('/inference-logs').json()
                print('Sanitized persisted logs:',logs)
                if len(logs['logs']) != 4 or any({'inputs','predictions','error_message'} & log.keys() for log in logs['logs']):
                    raise RuntimeError('Observation count or privacy contract failed')
                metrics_response = client.get('/metrics/observed-linear')
                metrics_response.raise_for_status()
                metrics = metrics_response.json()
                if (metrics['total_requests'],metrics['successful_requests'],metrics['failed_requests'],metrics['error_rate']) != (4,3,1,0.25):
                    raise RuntimeError('Aggregate counts or error rate are incorrect')
                if abs(metrics['throughput_requests_per_second'] - 4/300) > 1e-12:
                    raise RuntimeError('Observed throughput uses the wrong window')
                print('Five-minute metrics (success latency, nearest-rank percentiles):',metrics)
                print('Per-version metrics:',client.get('/metrics/observed-linear/versions').json())
                # A separate repository instance reads durable records, not counters.
                recreated = MetricsCollector(SQLiteInferenceLogRepository(sessionmaker(container.engine)))
                if recreated.get_metrics().total_requests != 4:
                    raise RuntimeError('Repository recreation lost observations')
                print('Persistence across repository recreation verified.')
        # All owned engine/cache resources are closed; reopen from disk.
        with ServiceContainer(settings) as reopened:
            if reopened.metrics_collector.get_metrics().total_requests != 4:
                raise RuntimeError('Observations did not survive container restart')
            print('Container restart persistence verified; no raw input/output fields stored.')
    print('Monitoring demo passed; temporary resources removed.')


if __name__ == '__main__':
    main()
