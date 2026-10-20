"""Revisioned routing snapshots, independent of persistence technology."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.domain.routing_config import RoutingConfig


@dataclass(frozen=True)
class RoutingSnapshot:
    config: RoutingConfig
    revision: int
    restored_from: int | None = None


class RoutingRepository(ABC):
    """Detached snapshots; current state and append-only history commit together.

    Initial revision is 1; duplicate creation raises RoutingConflictError.
    Missing current/history raises Routing[History]NotFoundError. Updates require
    a positive expected revision; stale writes raise RoutingConflictError.
    All referenced versions (including zero weights) must exist and remain
    protected from deletion by historical references. Other backend failures
    raise RepositoryError. Readiness is the application manager's responsibility.
    """

    @abstractmethod
    def get(self, model_name: str) -> RoutingSnapshot: ...

    @abstractmethod
    def create(self, config: RoutingConfig) -> RoutingSnapshot: ...

    @abstractmethod
    def update(self, config: RoutingConfig, *, expected_revision: int) -> RoutingSnapshot: ...

    @abstractmethod
    def history(self, model_name: str) -> list[RoutingSnapshot]:
        """Return history in revision order; missing current raises not-found."""

    @abstractmethod
    def get_revision(self, model_name: str, revision: int) -> RoutingSnapshot: ...

    @abstractmethod
    def restore(self, model_name: str, target_revision: int, *, expected_revision: int) -> RoutingSnapshot:
        """Append restored weights as a new revision, never mutate history."""
