"""Replaceable HTTP dependencies backed by the application-owned container."""

from typing import Annotated

from fastapi import Depends, Request

from app.core.container import ServiceContainer
from app.core.exceptions import InferenceUnavailableError
from app.services.deployment_manager import DeploymentManager
from app.services.inference_service import InferenceService


def get_container(request: Request) -> ServiceContainer:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise InferenceUnavailableError("Application lifespan has not initialized services")
    return container


def get_inference_service(
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> InferenceService:
    return container.inference_service


def get_deployment_manager(
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> DeploymentManager:
    return container.deployment_manager
