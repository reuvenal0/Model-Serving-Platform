"""Strict routing selection; no loading, prediction, or fallback."""

from dataclasses import dataclass

from app.core.exceptions import RoutingUnavailableError
from app.interfaces.deployment_repository import DeploymentRepository
from app.interfaces.model_repository import ModelRepository
from app.interfaces.routing_repository import RoutingRepository
from app.interfaces.routing_strategy import RoutingStrategy
from app.services.routing_eligibility import RoutingEligibility


@dataclass(frozen=True)
class RoutingSelection:
    version: str
    revision: int


class TrafficRouter:
    """All active destinations must have eligible deployment metadata.

    Readiness is checked by RoutingManager at configuration time and by the
    inference service for the selected runtime at admission. This component
    intentionally cannot discover latent runtime failures by loading models.
    No candidates are removed/renormalized. Selection uses one routing snapshot;
    later config changes apply to later requests, not a request already selected.
    """

    def __init__(self, routing: RoutingRepository, models: ModelRepository,
                 deployments: DeploymentRepository, strategy: RoutingStrategy) -> None:
        self._routing = routing
        self._models = models
        self._eligibility = RoutingEligibility(models, deployments)
        self._strategy = strategy

    def select(self, model_name: str) -> RoutingSelection:
        self._models.get_model(model_name)
        snapshot = self._routing.get(model_name)
        candidates = {version: weight for version, weight in snapshot.config.weights.items() if weight > 0}
        checked = [self._eligibility.require(model_name, version) for version in candidates]
        if not all(self._eligibility.unchanged(items) for items in checked):
            raise RoutingUnavailableError("Active routing deployment changed during selection")
        return RoutingSelection(self._strategy.select(candidates), snapshot.revision)
