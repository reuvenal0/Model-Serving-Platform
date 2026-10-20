"""Explicit application composition and ownership; nothing is built on import."""

from concurrent.futures import Future
import logging
from threading import Lock

from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.domain.enums import ModelFramework
from app.infrastructure.database.database import create_sqlite_engine, initialize_database
from app.infrastructure.database.sqlite_deployment_repository import SQLiteDeploymentRepository
from app.infrastructure.database.sqlite_model_repository import SQLiteModelRepository
from app.infrastructure.database.sqlite_routing_repository import SQLiteRoutingRepository
from app.infrastructure.database.sqlite_inference_log_repository import SQLiteInferenceLogRepository
from app.infrastructure.execution.onnx_model_loader import OnnxModelLoader
from app.infrastructure.execution.sklearn_model_loader import SklearnModelLoader
from app.infrastructure.storage.local_artifact_storage import LocalArtifactStorage
from app.services.deployment_health_checker import DeploymentHealthChecker
from app.services.deployment_manager import DeploymentManager
from app.services.inference_service import InferenceService
from app.services.metrics_collector import MetricsCollector
from app.services.model_loader_registry import ModelLoaderRegistry
from app.services.model_registry import ModelRegistry
from app.services.model_runtime_manager import ModelRuntimeManager
from app.services.routing_manager import RoutingManager
from app.services.traffic_router import TrafficRouter
from app.services.weighted_random_strategy import WeightedRandomStrategy
from app.interfaces.routing_strategy import RoutingStrategy
from app.core.exceptions import DeploymentConflictError, DeploymentRecoveryError, RepositoryError, StartupRecoveryError
from app.services.deployment_manager import ReconciliationReport
from app.services.metrics_collector import classify_error
from app.infrastructure.resources.process_memory import process_rss_bytes
from app.services.runtime_resource_monitor import RuntimeResourceMonitor


class ServiceContainer:
    """Own one engine/cache and share stateless factories, never sessions.

    Construct explicitly or let the FastAPI lifespan construct it. Externally
    injected containers remain caller-owned. Use a file SQLite database for the
    HTTP thread pool; the existing in-memory engine is connection/thread-local.
    """

    def __init__(self, settings: Settings, *, routing_strategy: RoutingStrategy | None = None) -> None:
        self.settings = settings
        self._closed = False
        self._startup_lock = Lock()
        self._startup_future: Future[ReconciliationReport] | None = None
        self.engine = create_sqlite_engine(settings.database_url)
        try:
            initialize_database(self.engine)
            self.session_factory = sessionmaker(self.engine)
            self.model_repository = SQLiteModelRepository(self.session_factory)
            self.deployment_repository = SQLiteDeploymentRepository(self.session_factory)
            self.routing_repository = SQLiteRoutingRepository(self.session_factory)
            self.inference_log_repository = SQLiteInferenceLogRepository(self.session_factory)
            self.metrics_collector = MetricsCollector(self.inference_log_repository)
            self.artifact_storage = LocalArtifactStorage(settings.artifact_storage_dir)
            self.model_registry = ModelRegistry(self.model_repository, self.artifact_storage)
            self.loader_registry = ModelLoaderRegistry({
                ModelFramework.SKLEARN: SklearnModelLoader(trusted=settings.trust_sklearn_artifacts),
                ModelFramework.ONNX: OnnxModelLoader(),
            })
            self.runtime_manager = ModelRuntimeManager(
                self.model_repository, self.artifact_storage, self.loader_registry,
                max_cache_size=settings.runtime_cache_max_size,
                max_artifact_bytes=settings.max_runtime_artifact_bytes,
            )
            self.health_checker = DeploymentHealthChecker()
            self.resource_monitor = RuntimeResourceMonitor(self.runtime_manager, process_rss_bytes)
            self.deployment_manager = DeploymentManager(
                self.deployment_repository, self.model_repository,
                self.runtime_manager, self.health_checker,
            )
            self.routing_strategy = routing_strategy if routing_strategy is not None else WeightedRandomStrategy()
            self.routing_manager = RoutingManager(
                self.routing_repository, self.model_repository,
                self.deployment_repository, self.deployment_manager,
            )
            self.traffic_router = TrafficRouter(
                self.routing_repository, self.model_repository,
                self.deployment_repository, self.routing_strategy,
            )
            self.inference_service = InferenceService(
                self.model_repository, self.deployment_repository,
                self.runtime_manager, self.health_checker,
                max_batch_size=settings.max_batch_size,
                max_tensor_dimensions=settings.max_tensor_dimensions,
                max_tensor_elements=settings.max_tensor_elements,
                traffic_router=self.traffic_router,
                metrics_collector=self.metrics_collector,
            )
        except BaseException:
            self.engine.dispose()
            raise

    @staticmethod
    def _fatal_recovery(error: Exception) -> bool:
        if isinstance(error, DeploymentRecoveryError):
            # A stale FAILED write must not overwrite a newer stop/scale. That
            # conflict is isolated, unlike a broken persistence backend.
            return (isinstance(error.operation_error, RepositoryError)
                    or not isinstance(error.persistence_error, DeploymentConflictError))
        return isinstance(error, RepositoryError)

    def reconcile_startup(self) -> ReconciliationReport:
        """Coalesce concurrent startup sweeps; wait without holding a lock.

        Sequential lifespans revalidate again, including caller-owned containers.
        Independent containers/processes are not coordinated. Explicit
        manager.reconcile() remains available for deliberate quiescent rechecks.
        """
        with self._startup_lock:
            if self._closed:
                raise StartupRecoveryError("Application resources are closed")
            owner = self._startup_future is None or self._startup_future.done()
            if owner:
                self._startup_future = Future()
            future = self._startup_future
        if not owner:
            return future.result()
        try:
            report = self.deployment_manager.reconcile()
            for failure in report.failures:
                logging.getLogger(__name__).error(
                    "startup_deployment_recovery_failed code=%s", classify_error(failure.error),
                )
            fatal = next((failure.error for failure in report.failures if
                          self._fatal_recovery(failure.error)), None)
            if fatal is not None:
                raise StartupRecoveryError("Recovery infrastructure is unavailable") from fatal
            logging.getLogger(__name__).info(
                "startup_reconciliation_complete checked=%d ready=%d failures=%d",
                len(report.health) + len(report.failures),
                sum(health.ready for health in report.health), len(report.failures),
            )
        except BaseException as error:
            # Release concurrent startup waiters even when the owner is interrupted.
            future.set_exception(error)
            raise
        future.set_result(report)
        return report

    def close(self) -> None:
        if not self._closed:
            try:
                self.runtime_manager.clear()
            finally:
                self.engine.dispose()
                self._closed = True

    def __enter__(self) -> "ServiceContainer":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
