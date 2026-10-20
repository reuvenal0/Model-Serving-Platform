"""Local-only monitoring reads through the injected collector, never SQL."""

from typing import Annotated
from dataclasses import replace

from fastapi import APIRouter, Depends, Query, Request
from app.api.security import require_operator
from pydantic import AwareDatetime

from app.api.dependencies import get_metrics_collector, get_runtime_resource_monitor
from app.api.routers.inference import ModelName
from app.api.schemas.inference import ErrorResponse
from app.api.schemas.monitoring import InferenceLogResponse, LogPageResponse, MetricsResponse, RuntimeResourcesResponse
from app.api.schemas.monitoring import AdmissionResourcesResponse
from app.interfaces.inference_log_repository import MAX_LOG_CURSOR, LogFilters
from app.services.metrics_collector import MetricsCollector
from app.services.runtime_resource_monitor import RuntimeResourceMonitor

router = APIRouter(dependencies=[Depends(require_operator)],
                   responses={status: {"model": ErrorResponse} for status in (401, 403, 422, 500, 503)})
Collector = Annotated[MetricsCollector, Depends(get_metrics_collector)]


@router.get("/runtime/admission", response_model=AdmissionResourcesResponse)
async def admission_resources(request: Request) -> dict[str, int | float]:
    """Capacity diagnostics only; rejected HTTP work is not an inference observation."""
    return request.app.state.inference_admission.snapshot()


def observation_filters(
    start: AwareDatetime | None = None, end: AwareDatetime | None = None,
    model_filter: Annotated[str | None, Query(alias="model_name", min_length=1, max_length=200)] = None,
    version: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
    success: bool | None = None, unknown_version: bool = False,
) -> LogFilters:
    return MetricsCollector.filters(start=start, end=end, model_name=model_filter, version=version,
                                    success=success, unknown_version=unknown_version)


Filters = Annotated[LogFilters, Depends(observation_filters)]


@router.get('/metrics', response_model=MetricsResponse)
def metrics(collector: Collector, filters: Filters) -> MetricsResponse:
    return MetricsResponse.model_validate(collector.get_metrics(filters))


@router.get('/metrics/{model_name}', response_model=MetricsResponse)
def model_metrics(model_name: ModelName, collector: Collector, filters: Filters) -> MetricsResponse:
    return MetricsResponse.model_validate(collector.get_metrics(replace(filters, model_name=model_name)))


@router.get('/metrics/{model_name}/versions', response_model=list[MetricsResponse])
def version_metrics(model_name: ModelName, collector: Collector, filters: Filters) -> list[MetricsResponse]:
    return [MetricsResponse.model_validate(item) for item in collector.get_version_metrics(replace(filters, model_name=model_name))]


@router.get('/inference-logs', response_model=LogPageResponse)
def inference_logs(collector: Collector, filters: Filters,
                   limit: Annotated[int, Query(ge=1, le=200)] = 50,
                   cursor: Annotated[int, Query(ge=0, le=MAX_LOG_CURSOR)] = 0) -> LogPageResponse:
    page = collector.query_logs(filters, limit=limit, cursor=cursor)
    return LogPageResponse(logs=[InferenceLogResponse(
        request_id=log.request_id, model_name=log.model_name, version=log.version,
        latency_ms=log.latency_ms, success=log.success, error_code=log.error_message,
        timestamp=log.timestamp,
    ) for log in page.logs], next_cursor=page.next_cursor, start=page.start, end=page.end)


@router.get('/runtime/resources', response_model=RuntimeResourcesResponse)
def runtime_resources(
    monitor: Annotated[RuntimeResourceMonitor, Depends(get_runtime_resource_monitor)],
) -> RuntimeResourcesResponse:
    """Local-development-only process RSS and cache counters; no model metadata."""
    return RuntimeResourcesResponse.model_validate(monitor.snapshot())
