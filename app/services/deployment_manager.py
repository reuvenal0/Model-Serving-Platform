"""Logical deployment orchestration without workers, endpoints, or routing."""

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any
from uuid import uuid4

from app.core.exceptions import (
    DeploymentConflictError, DeploymentRecoveryError, DomainValidationError,
)
from app.domain._validation import nonempty, utc_now
from app.domain.deployment import Deployment
from app.domain.enums import DeploymentStatus, ModelVersionStatus
from app.domain.resource_config import ResourceConfig
from app.interfaces.deployment_repository import DeploymentRepository, DeploymentSnapshot
from app.interfaces.model_repository import ModelRepository
from app.services.deployment_health_checker import (
    DeploymentHealthChecker, NO_SMOKE_INPUT, ReadinessCheck,
)
from app.services.model_runtime_manager import ModelRuntimeManager


@dataclass(frozen=True)
class DeploymentHealth:
    deployment_id: str
    status: DeploymentStatus
    ready: bool
    kind: str
    checked_at: datetime


@dataclass(frozen=True)
class ReconciliationFailure:
    deployment_id: str
    error: Exception


@dataclass(frozen=True)
class ReconciliationReport:
    health: tuple[DeploymentHealth, ...]
    failures: tuple[ReconciliationFailure, ...]


class DeploymentManager:
    """Persist lifecycle state; runtimes are shared by immutable model version.

    No global lock or transaction spans loading/prediction. Revisions guard
    completion and failure updates; a conflict must be re-read, never blindly
    retried. Persisted status alone is not proof of current process readiness.
    """

    def __init__(
        self, deployments: DeploymentRepository, models: ModelRepository,
        runtimes: ModelRuntimeManager, health_checker: DeploymentHealthChecker,
    ) -> None:
        self._deployments = deployments
        self._models = models
        self._runtimes = runtimes
        self._health_checker = health_checker

    def _record_failure(self, snapshot: DeploymentSnapshot, error: Exception) -> None:
        failed = replace(snapshot.deployment)
        failed.fail()
        try:
            self._deployments.update_deployment(failed, expected_revision=snapshot.revision)
        except Exception as persistence_error:
            raise DeploymentRecoveryError(failed.id, error, persistence_error) from persistence_error

    def _readiness(self, deployment: Deployment, smoke_inputs: Any) -> ReadinessCheck:
        runtime = self._runtimes.get_runtime(deployment.model_name, deployment.version)
        return self._health_checker.check(runtime, smoke_inputs=smoke_inputs)

    def deploy(
        self, model_name: str, version: str, resources: ResourceConfig, *,
        replicas: int = 1, deployment_id: str | None = None,
        smoke_inputs: Any = NO_SMOKE_INPUT,
    ) -> Deployment:
        deployment = Deployment(
            id=uuid4().hex if deployment_id is None else deployment_id,
            model_name=model_name, version=version, replicas=replicas, resources=resources,
        )
        self._models.get_model(model_name)
        metadata = self._models.get_version(model_name, version)
        if metadata.status is not ModelVersionStatus.REGISTERED:
            raise DomainValidationError("Only REGISTERED versions are eligible for deployment")
        snapshot = self._deployments.save_deployment(deployment)
        try:
            self._readiness(snapshot.deployment, smoke_inputs)
        except Exception as error:
            self._record_failure(snapshot, error)
            raise
        running = replace(snapshot.deployment)
        running.activate()
        try:
            return self._deployments.update_deployment(
                running, expected_revision=snapshot.revision,
            ).deployment
        except DeploymentConflictError:
            # Someone stopped/scaled/changed it while loading. Never restore RUNNING.
            raise
        except Exception as error:
            self._record_failure(snapshot, error)
            raise

    def undeploy(self, deployment_id: str) -> Deployment:
        nonempty(deployment_id, "deployment_id")
        snapshot = self._deployments.get_deployment(deployment_id)
        stopped = replace(snapshot.deployment)
        stopped.stop()  # Repeated stopping intentionally remains illegal.
        return self._deployments.update_deployment(
            stopped, expected_revision=snapshot.revision,
        ).deployment

    def scale(self, deployment_id: str, replicas: int) -> Deployment:
        nonempty(deployment_id, "deployment_id")
        snapshot = self._deployments.get_deployment(deployment_id)
        scaled = replace(snapshot.deployment)
        scaled.scale(replicas)
        return self._deployments.update_deployment(
            scaled, expected_revision=snapshot.revision,
        ).deployment

    def get_deployment(self, deployment_id: str) -> Deployment:
        nonempty(deployment_id, "deployment_id")
        return self._deployments.get_deployment(deployment_id).deployment

    def list_deployments(self, model_name: str | None = None) -> list[Deployment]:
        if model_name is not None:
            nonempty(model_name, "model_name")
        return [snapshot.deployment for snapshot in self._deployments.list_deployments(model_name)]

    @staticmethod
    def _not_ready(deployment: Deployment, kind: str = "not_running") -> DeploymentHealth:
        return DeploymentHealth(deployment.id, deployment.status, False, kind, utc_now())

    def check_health(
        self, deployment_id: str, *, smoke_inputs: Any = NO_SMOKE_INPUT,
    ) -> DeploymentHealth:
        nonempty(deployment_id, "deployment_id")
        snapshot = self._deployments.get_deployment(deployment_id)
        deployment = snapshot.deployment
        if deployment.status is not DeploymentStatus.RUNNING:
            return self._not_ready(deployment)
        try:
            check = self._readiness(deployment, smoke_inputs)
        except Exception as error:
            self._record_failure(snapshot, error)
            raise
        current = self._deployments.get_deployment(deployment_id)
        if current.revision != snapshot.revision:
            raise DeploymentConflictError("Deployment changed during its health check; re-read state")
        return DeploymentHealth(deployment.id, deployment.status, True, check.kind, check.checked_at)

    def reconcile(self) -> ReconciliationReport:
        """Explicit quiescent startup check; call before advertising readiness.

        Revalidate RUNNING records using load checks (lazily reload if evicted).
        Mark interrupted DEPLOYING records FAILED without restarting them. All
        errors are returned explicitly; inspect failures before accepting work.
        This is not a background recovery loop or cross-process startup lease.
        """
        health = []
        failures = []
        for snapshot in self._deployments.list_deployments():
            deployment = snapshot.deployment
            try:
                if deployment.status is DeploymentStatus.RUNNING:
                    try:
                        self._runtimes.verify_available(deployment.model_name, deployment.version)
                    except Exception as error:
                        self._runtimes.invalidate(deployment.model_name, deployment.version)
                        self._record_failure(snapshot, error)
                        raise
                    health.append(self.check_health(deployment.id))
                elif deployment.status is DeploymentStatus.DEPLOYING:
                    interrupted = replace(deployment)
                    interrupted.fail()
                    result = self._deployments.update_deployment(
                        interrupted, expected_revision=snapshot.revision,
                    )
                    health.append(self._not_ready(result.deployment, "interrupted"))
                else:
                    health.append(self._not_ready(deployment))
            except Exception as error:
                failures.append(ReconciliationFailure(deployment.id, error))
        return ReconciliationReport(tuple(health), tuple(failures))
