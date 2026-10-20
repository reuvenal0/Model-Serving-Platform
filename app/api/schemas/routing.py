"""Local routing management request and response contracts."""

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictInt

from app.interfaces.routing_repository import RoutingSnapshot


class RoutingUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    weights: dict[str, StrictInt]
    expected_revision: int = Field(ge=0)


class RoutingRollbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    target_revision: int = Field(ge=1)
    expected_revision: int = Field(ge=1)


class RoutingResponse(BaseModel):
    model_name: str
    weights: dict[str, int]
    revision: int
    updated_at: AwareDatetime
    restored_from: int | None

    @classmethod
    def from_snapshot(cls, snapshot: RoutingSnapshot) -> "RoutingResponse":
        return cls(model_name=snapshot.config.model_name, weights=dict(snapshot.config.weights),
                   revision=snapshot.revision, updated_at=snapshot.config.updated_at,
                   restored_from=snapshot.restored_from)
