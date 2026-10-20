"""Repeatable logical deployment demonstration; no API or worker processes."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any

from sklearn.linear_model import LinearRegression
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

from app.core.exceptions import InvalidDeploymentStateError
from app.domain.enums import ModelFramework
from app.domain.resource_config import ResourceConfig
from app.infrastructure.database.database import create_sqlite_engine, initialize_database
from app.infrastructure.database.sqlite_deployment_repository import SQLiteDeploymentRepository
from app.infrastructure.database.sqlite_model_repository import SQLiteModelRepository
from app.infrastructure.execution.sklearn_model_loader import SklearnModelLoader
from app.infrastructure.storage.local_artifact_storage import LocalArtifactStorage
from app.interfaces.loaded_model import LoadedModel
from app.services.deployment_health_checker import (
    DeploymentHealthChecker, NO_SMOKE_INPUT, ReadinessCheck,
)
from app.services.deployment_manager import DeploymentManager
from app.services.model_loader_registry import ModelLoaderRegistry
from app.services.model_registry import ModelRegistry
from app.services.model_runtime_manager import ModelRuntimeManager


class ObservableReadinessChecker(DeploymentHealthChecker):
    """Pause the demonstration so the main thread can inspect DEPLOYING."""

    def __init__(self) -> None:
        self.entered = Event()
        self.proceed = Event()

    def check(
        self, runtime: LoadedModel, *, smoke_inputs: Any = NO_SMOKE_INPUT,
    ) -> ReadinessCheck:
        self.entered.set()
        if not self.proceed.wait(10):
            raise RuntimeError("Demo readiness observation timed out")
        return super().check(runtime, smoke_inputs=smoke_inputs)


def main() -> None:
    estimator = LinearRegression().fit([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0])
    trusted_artifact = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
    with TemporaryDirectory(prefix="deployment-demo-") as directory:
        root = Path(directory)
        url = URL.create("sqlite", database=str(root / "metadata.db")).render_as_string()
        engine = create_sqlite_engine(url)
        try:
            initialize_database(engine)
            sessions = sessionmaker(engine)
            models = SQLiteModelRepository(sessions)
            storage = LocalArtifactStorage(root / "artifacts")
            registry = ModelRegistry(models, storage)
            registry.register_model("linear-demo")
            registry.register_version("linear-demo", "v1", ModelFramework.SKLEARN, trusted_artifact)
            deployments = SQLiteDeploymentRepository(sessions)
            runtimes = ModelRuntimeManager(models, storage, ModelLoaderRegistry({
                ModelFramework.SKLEARN: SklearnModelLoader(trusted=True),
            }))
            checker = ObservableReadinessChecker()
            manager = DeploymentManager(deployments, models, runtimes, checker)
            identifier = "demo-deployment"
            # This thread makes the synchronous lifecycle observable; replicas
            # below remain metadata, not execution threads or processes.
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(
                    manager.deploy, "linear-demo", "v1", ResourceConfig(1, 128),
                    deployment_id=identifier, smoke_inputs=[[3.0]],
                )
                try:
                    if not checker.entered.wait(10):
                        raise RuntimeError("Demo deployment did not reach readiness")
                    print("Persisted before readiness:", manager.get_deployment(identifier).status.value)
                finally:
                    checker.proceed.set()
                deployed = pending.result(timeout=10)
            print("After readiness:", deployed.status.value)
            health = manager.check_health(identifier)
            print(f"Ready: {health.ready} ({health.kind} check)")
            scaled = manager.scale(identifier, 3)
            print(f"Logical replicas: {deployed.replicas} -> {scaled.replicas}; no extra workers created")
            persisted = SQLiteDeploymentRepository(sessions).get_deployment(identifier)
            print(f"Persisted deployment: {persisted.deployment.id}, status={persisted.deployment.status.value}, "
                  f"replicas={persisted.deployment.replicas}, revision={persisted.revision}")
            stopped = manager.undeploy(identifier)
            print("Final status:", stopped.status.value)
            print("Ready after stop:", manager.check_health(identifier).ready)
            try:
                manager.scale(identifier, 4)
            except InvalidDeploymentStateError as error:
                print("Invalid transition rejected:", str(error))
            else:
                raise RuntimeError("Scaling a stopped deployment should have failed")
            print("Deployment demo passed; temporary data will be removed.")
        finally:
            engine.dispose()


if __name__ == "__main__":
    main()
