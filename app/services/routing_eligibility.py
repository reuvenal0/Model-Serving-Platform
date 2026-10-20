"""Shared metadata eligibility checks; never load or execute a model."""

from app.core.exceptions import DeploymentNotFoundError, RoutingUnavailableError
from app.domain.enums import DeploymentStatus, ModelVersionStatus
from app.interfaces.deployment_repository import DeploymentRepository, DeploymentSnapshot
from app.interfaces.model_repository import ModelRepository


class RoutingEligibility:
    def __init__(self, models: ModelRepository, deployments: DeploymentRepository) -> None:
        self._models = models
        self._deployments = deployments

    def require(self, model_name: str, version: str) -> list[DeploymentSnapshot]:
        metadata = self._models.get_version(model_name, version)
        if metadata.status is not ModelVersionStatus.REGISTERED:
            raise RoutingUnavailableError("Active routing version is not REGISTERED")
        candidates = [item for item in self._deployments.list_deployments(model_name)
                      if item.deployment.version == version and item.deployment.status is DeploymentStatus.RUNNING]
        if not candidates:
            raise RoutingUnavailableError("Active routing destination has no RUNNING deployment")
        return candidates

    def unchanged(self, snapshots: list[DeploymentSnapshot]) -> bool:
        for snapshot in snapshots:
            try:
                current = self._deployments.get_deployment(snapshot.deployment.id)
            except DeploymentNotFoundError:
                continue
            if (current.revision == snapshot.revision
                    and current.deployment.model_name == snapshot.deployment.model_name
                    and current.deployment.version == snapshot.deployment.version
                    and current.deployment.status is DeploymentStatus.RUNNING):
                return True
        return False
