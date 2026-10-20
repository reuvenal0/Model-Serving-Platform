"""Monitoring DTOs expose sanitized metadata only."""

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class MetricsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, allow_inf_nan=False)
    model_name: str | None
    version: str | None
    start: AwareDatetime
    end: AwareDatetime
    total_requests: int = Field(ge=0)
    successful_requests: int = Field(ge=0)
    failed_requests: int = Field(ge=0)
    error_rate: float = Field(ge=0, le=1)
    throughput_requests_per_second: float = Field(ge=0)
    average_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    p99_latency_ms: float | None


class InferenceLogResponse(BaseModel):
    request_id: str
    model_name: str | None
    version: str | None
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    success: bool
    error_code: str | None
    timestamp: AwareDatetime


class LogPageResponse(BaseModel):
    logs: list[InferenceLogResponse]
    next_cursor: int | None
    start: AwareDatetime
    end: AwareDatetime
