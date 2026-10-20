"""HTTP inference adapters admit bounded work before native thread execution."""

from typing import Annotated

from fastapi import APIRouter, Depends, Path, Request
from app.api.security import require_admin

from app.api.dependencies import get_deployment_manager, get_inference_service
from app.api.schemas.inference import (
    DeploymentHealthResponse, ErrorResponse, LivenessResponse, PredictionRequest, PredictionResponse,
)
from app.services.deployment_manager import DeploymentManager
from app.services.inference_service import InferenceService


router = APIRouter(responses={status: {"model": ErrorResponse} for status in (404, 409, 413, 422, 500, 503)})
ModelName = Annotated[str, Path(min_length=1, max_length=200)]


@router.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict(
    request: Request,
    model_name: ModelName, body: PredictionRequest,
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> PredictionResponse:
    return await request.app.state.inference_admission.run(
        lambda: PredictionResponse.model_validate(service.predict(model_name, body.version, body.inputs)))


@router.post("/predict/{model_name}/batch", response_model=PredictionResponse)
async def batch_predict(
    request: Request,
    model_name: ModelName, body: PredictionRequest,
    service: Annotated[InferenceService, Depends(get_inference_service)],
) -> PredictionResponse:
    """Explicit batch alias: one bounded tensor invocation, preserving sample order."""
    return await request.app.state.inference_admission.run(
        lambda: PredictionResponse.model_validate(service.batch_predict(model_name, body.version, body.inputs)))


@router.get("/health", response_model=LivenessResponse)
def health() -> LivenessResponse:
    """API liveness only, independent of deployed-model readiness."""
    return LivenessResponse()


@router.get("/deployments/{deployment_id}/health", response_model=DeploymentHealthResponse,
            dependencies=[Depends(require_admin)])
def deployment_health(
    deployment_id: Annotated[str, Path(min_length=1, max_length=200)],
    manager: Annotated[DeploymentManager, Depends(get_deployment_manager)],
) -> DeploymentHealthResponse:
    return DeploymentHealthResponse.model_validate(manager.check_health(deployment_id))
