"""A registered model identity, independent of its versions."""

from dataclasses import dataclass, field
from datetime import datetime

from app.domain._validation import nonempty, utc_now, utc_timestamp


@dataclass
class Model:
    name: str
    description: str = ""
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        nonempty(self.name, "name")
        self.created_at = utc_timestamp(self.created_at, "created_at")
        self.updated_at = utc_timestamp(self.updated_at, "updated_at")
