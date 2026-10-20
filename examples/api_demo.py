"""Real HTTP inference using temporary resources and in-process TestClient."""

from pathlib import Path
import pickle
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sqlalchemy.engine import URL

from app.config import Settings
from app.core.container import ServiceContainer
from app.domain.enums import ModelFramework
from app.domain.resource_config import ResourceConfig
from app.main import create_app


def main() -> None:
    # The only Pickle bytes used here are trained/serialized locally by this demo.
    estimator = LinearRegression().fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0])
    artifact = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
    with TemporaryDirectory(prefix="api-demo-") as directory:
        root = Path(directory)
        settings = Settings(
            database_url=URL.create("sqlite", database=str(root / "metadata.db")).render_as_string(),
            artifact_storage_dir=root / "artifacts", trust_sklearn_artifacts=True,
        )
        app = create_app(settings)
        with TestClient(app) as client:
            container: ServiceContainer = app.state.container
            print("API startup liveness:", client.get("/health").json())
            container.model_registry.register_model("linear-demo")
            container.model_registry.register_version(
                "linear-demo", "v1", ModelFramework.SKLEARN, artifact,
                input_schema={"inference": {"shape": [None, 1], "dtype": "number"}},
                output_schema={"inference": {"shape": [None], "dtype": "number"}},
            )
            deployment = container.deployment_manager.deploy(
                "linear-demo", "v1", ResourceConfig(1, 128), deployment_id="demo-deployment",
            )
            health = client.get(f"/deployments/{deployment.id}/health")
            print("Deployment health:", health.json())
            prediction = client.post("/predict/linear-demo", json={"version": "v1", "inputs": [[3.0]]})
            prediction.raise_for_status()
            body = prediction.json()
            if abs(body["predictions"][0] - 7.0) > 1e-8 or body["latency_ms"] < 0:
                raise RuntimeError("Prediction result or metadata is incorrect")
            print("HTTP prediction:", body)
            batch = client.post("/predict/linear-demo/batch", json={"version": "v1", "inputs": [[4.0], [3.0]]})
            batch.raise_for_status()
            if len(batch.json()["predictions"]) != 2 or any(
                abs(actual - expected) > 1e-8 for actual, expected in zip(batch.json()["predictions"], [9, 7])
            ):
                raise RuntimeError("Batch output order is incorrect")
            print("HTTP batch:", batch.json())
            print("Runtime loads:", container.runtime_manager.cache_info().loads)
            container.deployment_manager.undeploy(deployment.id)
            print("Stopped readiness:", client.get(f"/deployments/{deployment.id}/health").json()["ready"])
            rejected = client.post("/predict/linear-demo", json={"version": "v1", "inputs": [[3.0]]})
            if rejected.status_code != 503:
                raise RuntimeError("Inference after undeploy must be rejected")
            print("After undeploy:", rejected.status_code, rejected.json())
            print("Process remains live:", client.get("/health").json())
        if app.state.container is not None or container.runtime_manager.cache_info().current_size != 0:
            raise RuntimeError("Application did not release its owned resources")
        print("API shutdown verified; temporary database and artifacts will be removed.")


if __name__ == "__main__":
    main()
