"""Small management DTOs; artifact locations never leave the service layer."""

from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, StringConstraints

from app.domain.enums import DeploymentStatus, ModelFramework, ModelVersionStatus

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: Identifier
    description: str = Field(default="", max_length=4000)


class ModelResponse(ModelRequest):
    model_config = ConfigDict(from_attributes=True)
    created_at: AwareDatetime
    updated_at: AwareDatetime


class VersionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Identifier
    onnx_base64: str = Field(min_length=1)
    input_schema: dict[str, JsonValue] = Field(default_factory=dict)
    output_schema: dict[str, JsonValue] = Field(default_factory=dict)


class VersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    model_name: str
    version: str
    framework: ModelFramework
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]
    status: ModelVersionStatus
    created_at: AwareDatetime


class Resources(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, from_attributes=True)
    cpu: float = Field(gt=0, allow_inf_nan=False)
    memory_mb: int = Field(gt=0)


class DeploymentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model_name: Identifier
    version: Identifier
    resources: Resources
    replicas: int = Field(default=1, ge=0)
    deployment_id: Identifier | None = None


class DeploymentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    model_name: str
    version: str
    resources: Resources
    replicas: int
    status: DeploymentStatus
    created_at: AwareDatetime
    updated_at: AwareDatetime
