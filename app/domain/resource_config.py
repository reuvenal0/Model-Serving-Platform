"""Infrastructure-independent computational resource requirements."""

from dataclasses import dataclass

from app.core.exceptions import DomainValidationError
from app.domain._validation import finite_number


@dataclass(frozen=True)
class ResourceConfig:
    cpu: float
    memory_mb: int

    def __post_init__(self) -> None:
        finite_number(self.cpu, "cpu")
        if self.cpu <= 0:
            raise DomainValidationError("cpu must be strictly positive")
        if type(self.memory_mb) is not int or self.memory_mb <= 0:
            raise DomainValidationError("memory_mb must be a strictly positive integer")
