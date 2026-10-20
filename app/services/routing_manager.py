"""Validate live targets before committing a versioned routing allocation."""

from collections.abc import Mapping

from app.core.exceptions import (
    DeploymentConflictError, DomainValidationError, RoutingConflictError,
    RoutingNotFoundError, RoutingUnavailableError,
)
from app.domain.routing_config import RoutingConfig
from app.interfaces.deployment_repository import DeploymentRepository
from app.interfaces.model_repository import ModelRepository
from app.interfaces.routing_repository import RoutingRepository, RoutingSnapshot
from app.services.deployment_manager import DeploymentManager
from app.services.routing_eligibility import RoutingEligibility


class RoutingManager:
    """Readiness and routing CAS are separate operations, not a shared lease.

    All positive targets must pass the existing load-readiness contract. Zero
    targets must exist but may be stopped, failed or INVALID. Revision rechecks
    detect observed changes during checks; a stop immediately after the final
    check can still race the commit. TrafficRouter and inference recheck later.
    """

    def __init__(self, routing: RoutingRepository, models: ModelRepository,
                 deployments: DeploymentRepository, manager: DeploymentManager) -> None:
        self._routing = routing
        self._models = models
        self._manager = manager
        self._eligibility = RoutingEligibility(models, deployments)

    def get_routing_config(self, model_name: str) -> RoutingSnapshot:
        self._models.get_model(model_name)
        return self._routing.get(model_name)

    def get_routing_history(self, model_name: str) -> list[RoutingSnapshot]:
        self._models.get_model(model_name)
        return self._routing.history(model_name)

    @staticmethod
    def _expected(expected_revision: int) -> None:
        if type(expected_revision) is not int or expected_revision < 0:
            raise DomainValidationError("expected_revision must be a nonnegative integer")

    def _check_revision(self, model_name: str, expected_revision: int) -> None:
        self._expected(expected_revision)
        try:
            current = self._routing.get(model_name)
        except RoutingNotFoundError:
            if expected_revision != 0:
                raise
        else:
            if expected_revision != current.revision:
                raise RoutingConflictError("Routing changed; re-read before updating")

    def _validate_targets(self, config: RoutingConfig) -> None:
        checked = []
        for version, weight in config.weights.items():
            self._models.get_version(config.model_name, version)
            if weight == 0:
                continue
            candidates = self._eligibility.require(config.model_name, version)
            for candidate in candidates:
                health = self._manager.check_health(candidate.deployment.id)
                if health.ready and self._eligibility.unchanged([candidate]):
                    checked.append([candidate])
                    break
            else:
                raise RoutingUnavailableError("Routing destination did not pass readiness")
        if not all(self._eligibility.unchanged(items) for items in checked):
            raise DeploymentConflictError("Deployment changed while validating routing")

    def configure_routing(self, model_name: str, weights: Mapping[str, int], expected_revision: int) -> RoutingSnapshot:
        config = RoutingConfig(model_name, weights)
        self._models.get_model(model_name)
        self._check_revision(model_name, expected_revision)
        self._validate_targets(config)
        if expected_revision == 0:
            return self._routing.create(config)
        return self._routing.update(config, expected_revision=expected_revision)

    def rollback(self, model_name: str, target_revision: int, expected_revision: int) -> RoutingSnapshot:
        self._models.get_model(model_name)
        historical = self._routing.get_revision(model_name, target_revision)
        self._expected(expected_revision)
        current = self._routing.get(model_name)
        if current.revision != expected_revision:
            raise RoutingConflictError("Routing changed; re-read before rollback")
        self._validate_targets(RoutingConfig(model_name, historical.config.weights))
        return self._routing.restore(model_name, target_revision, expected_revision=expected_revision)
