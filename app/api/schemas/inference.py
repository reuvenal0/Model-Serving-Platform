"""Strict HTTP DTOs; model-specific tensor validation belongs to the service."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, field_validator

from app.domain.enums import DeploymentStatus


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    version: str | None = Field(default=None, min_length=1, max_length=200)
    inputs: list[JsonValue] | dict[str, list[JsonValue]]

    @field_validator("version")
    @classmethod
    def nonblank_version(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Version cannot be blank")
        return value


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_name: str
    version: str
    predictions: JsonValue
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class DeploymentHealthResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    deployment_id: str
    status: DeploymentStatus
    ready: bool
    kind: str
    checked_at: AwareDatetime


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
