"""Version selection from caller-verified weighted candidates."""

from abc import ABC, abstractmethod
from collections.abc import Mapping


class RoutingStrategy(ABC):
    @abstractmethod
    def select(self, candidates: Mapping[str, int]) -> str:
        """Select a positive-weight version, preserving relative probabilities.

        Weights need not sum to 100, but must be integer percentages, with at
        least one positive value. Invalid candidates raise DomainValidationError.
        """
