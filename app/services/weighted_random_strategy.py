"""Weighted selection without process-global random state."""

from collections.abc import Mapping
from random import Random
from threading import Lock

from app.core.exceptions import DomainValidationError
from app.domain._validation import nonempty
from app.interfaces.routing_strategy import RoutingStrategy


class WeightedRandomStrategy(RoutingStrategy):
    def __init__(self, rng: Random | None = None) -> None:
        self._rng = rng if rng is not None else Random()
        self._lock = Lock()

    def select(self, candidates: Mapping[str, int]) -> str:
        if not isinstance(candidates, Mapping) or not candidates:
            raise DomainValidationError("Candidates must be a nonempty mapping")
        entries = list(candidates.items())
        for version, weight in entries:
            nonempty(version, "version")
            if type(weight) is not int or not 0 <= weight <= 100:
                raise DomainValidationError("Candidate weights must be integer percentages")
        total = sum(weight for _, weight in entries)
        if total == 0:
            raise DomainValidationError("At least one candidate must have positive weight")
        with self._lock:
            ticket = self._rng.randrange(total)
        for version, weight in entries:
            if ticket < weight:
                return version
            ticket -= weight
        raise RuntimeError("Random generator returned an out-of-range ticket")
