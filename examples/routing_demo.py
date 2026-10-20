"""Temporary two-version canary, rollback, conflicts, and strict availability."""

from collections import Counter
from pathlib import Path
import pickle
from random import Random
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from app.config import Settings
from app.core.container import ServiceContainer
from app.domain.enums import ModelFramework
from app.domain.resource_config import ResourceConfig
from app.main import create_app
from app.services.weighted_random_strategy import WeightedRandomStrategy


def main() -> None:
    with TemporaryDirectory(prefix='routing-demo-') as directory:
        root = Path(directory)
        settings = Settings(database_url=f"sqlite:///{root / 'metadata.db'}",
                            artifact_storage_dir=root / 'artifacts', trust_sklearn_artifacts=True)
        with ServiceContainer(settings, routing_strategy=WeightedRandomStrategy(Random(42))) as container:
            registry = container.model_registry
            registry.register_model('routing-demo')
            for version, slope in [('v1', 2), ('v2', 3)]:
                estimator = LinearRegression().fit([[1.0], [2.0], [3.0]],
                                                    [slope + 1, slope*2 + 1, slope*3 + 1])
                registry.register_version('routing-demo', version, ModelFramework.SKLEARN,
                                          pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL))
                container.deployment_manager.deploy('routing-demo', version, ResourceConfig(1, 128), deployment_id=version)
            with TestClient(create_app(container=container)) as client:
                def allocation(weights, expected):
                    response = client.put('/routing/routing-demo', json={'weights': weights, 'expected_revision': expected})
                    response.raise_for_status()
                    print('Allocation:', response.json())
                    return response.json()['revision']

                def sample(count):
                    counts = Counter()
                    for _ in range(count):
                        response = client.post('/predict/routing-demo', json={'inputs': [[4.0]]})
                        response.raise_for_status()
                        body = response.json()
                        expected = 9.0 if body['version'] == 'v1' else 13.0
                        if abs(body['predictions'][0] - expected) > 1e-8:
                            raise RuntimeError('Incorrect model-specific prediction')
                        counts[body['version']] += 1
                    print('Observed destinations:', dict(counts))
                    return counts

                revision = allocation({'v1': 90, 'v2': 10}, 0)
                counts = sample(100)
                if not all(counts[version] > 0 for version in ('v1', 'v2')):
                    raise RuntimeError('Seeded sample did not exercise both versions')
                allocation({'v1': 0, 'v2': 100}, revision)
                if sample(5) != Counter(v2=5):
                    raise RuntimeError('100% allocation failed')
                restored = client.post('/routing/routing-demo/rollback', json={'target_revision': 1, 'expected_revision': 2})
                restored.raise_for_status()
                print('Rollback:', restored.json())
                if restored.json()['weights'] != {'v1': 90, 'v2': 10} or restored.json()['revision'] != 3:
                    raise RuntimeError('Rollback did not append restored weights')
                sample(20)
                print('Immutable revision history:', client.get('/routing/routing-demo/history').json())
                stale = client.put('/routing/routing-demo', json={'weights': {'v1': 100}, 'expected_revision': 2})
                if stale.status_code != 409:
                    raise RuntimeError('Stale update was accepted')
                print('Stale update:', stale.status_code, stale.json())
                container.deployment_manager.undeploy('v2')
                unavailable = client.post('/predict/routing-demo', json={'inputs': [[4.0]]})
                if unavailable.status_code != 503:
                    raise RuntimeError('Unavailable active destination was silently rerouted')
                print('Unavailable allocation:', unavailable.status_code, unavailable.json())
                explicit = client.post('/predict/routing-demo', json={'version': 'v1', 'inputs': [[4.0]]})
                explicit.raise_for_status()
                print('Explicit-version compatibility:', explicit.json())
        print('Routing demo passed; owned resources closed and temporary data removed.')


if __name__ == '__main__':
    main()
