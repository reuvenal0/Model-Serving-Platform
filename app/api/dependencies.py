"""Replaceable HTTP dependencies backed by the application-owned container."""

from typing import Annotated

from fastapi import Depends, Request

from app.core.container import ServiceContainer
from app.core.exceptions import InferenceUnavailableError
from app.services.deployment_manager import DeploymentManager
from app.services.inference_service import InferenceService
from app.services.routing_manager import RoutingManager
from app.services.metrics_collector import MetricsCollector


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


def get_routing_manager(
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> RoutingManager:
    return container.routing_manager


def get_metrics_collector(
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> MetricsCollector:
    return container.metrics_collector
