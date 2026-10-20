"""Loopback-only management adapters over existing application services."""

import base64
import binascii
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.dependencies import get_deployment_manager, get_model_registry
from app.api.schemas.management import (
    DeploymentRequest, DeploymentResponse, ModelRequest, ModelResponse,
    VersionRequest, VersionResponse,
)
from app.api.security import require_admin, require_operator
from app.domain.enums import ModelFramework
from app.domain.resource_config import ResourceConfig
from app.services.deployment_manager import DeploymentManager
from app.services.model_registry import ModelRegistry

router = APIRouter(tags=["management"], dependencies=[Depends(require_operator)])
Registry = Annotated[ModelRegistry, Depends(get_model_registry)]
Manager = Annotated[DeploymentManager, Depends(get_deployment_manager)]


@router.post("/models", response_model=ModelResponse, status_code=201,
             dependencies=[Depends(require_admin)])
def register_model(body: ModelRequest, registry: Registry):
    return registry.register_model(body.name, body.description)


@router.get("/models", response_model=list[ModelResponse])
def list_models(registry: Registry):
    return registry.list_models()


@router.get("/models/{name}/versions", response_model=list[VersionResponse])
def list_versions(name: str, registry: Registry):
    registry.get_model(name)
    return registry.list_versions(name)


@router.post("/models/{name}/versions", response_model=VersionResponse, status_code=201,
             dependencies=[Depends(require_admin)])
def register_version(name: str, body: VersionRequest, request: Request, registry: Registry):
    # The existing middleware bounds the entire body before JSON parsing. Check
    # encoded length before allocating decoded bytes, then independently check bytes.
    maximum = request.app.state.security_settings.max_upload_size_bytes
    if len(body.onnx_base64) > 4 * ((maximum + 2) // 3):
        raise HTTPException(413)
    try:
        content = base64.b64decode(body.onnx_base64, validate=True)
    except (ValueError, binascii.Error):
        raise HTTPException(422) from None
    if not content:
        raise HTTPException(422)
    if len(content) > maximum:
        raise HTTPException(413)
    # Registration stores bytes only. Deployment performs ONNX load-readiness;
    # uploaded bytes can never select the sklearn/Pickle loader.
    return registry.register_version(
        name, body.version, ModelFramework.ONNX, content,
        input_schema=body.input_schema, output_schema=body.output_schema,
    )


@router.post("/deployments", response_model=DeploymentResponse, status_code=201,
             dependencies=[Depends(require_admin)])
def deploy(body: DeploymentRequest, manager: Manager):
    return manager.deploy(
        body.model_name, body.version,
        ResourceConfig(body.resources.cpu, body.resources.memory_mb),
        replicas=body.replicas, deployment_id=body.deployment_id,
    )


@router.get("/deployments", response_model=list[DeploymentResponse])
def list_deployments(manager: Manager):
    return manager.list_deployments()


@router.post("/deployments/{deployment_id}/stop", response_model=DeploymentResponse,
             dependencies=[Depends(require_admin)])
def stop(deployment_id: str, manager: Manager):
    return manager.undeploy(deployment_id)
