"""Unauthenticated routing controls for local development only."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.routing import APIRoute

from app.api.dependencies import get_routing_manager
from app.api.routers.inference import ModelName
from app.api.schemas.inference import ErrorResponse
from app.api.schemas.routing import RoutingResponse, RoutingRollbackRequest, RoutingUpdateRequest
from app.core.exceptions import DomainValidationError
from app.services.routing_manager import RoutingManager


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DomainValidationError("Duplicate routing JSON keys are invalid")
        result[key] = value
    return result


class RoutingJSONRoute(APIRoute):
    """Reject duplicate keys before Starlette's JSON parser can discard them."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def checked(request: Request):
            if request.method in ("PUT", "POST"):
                # The application body-limit middleware bounds this cached read.
                try:
                    json.loads(await request.body(), object_pairs_hook=_unique_object)
                except (ValueError, UnicodeError, RecursionError) as error:
                    raise DomainValidationError("Routing JSON is invalid") from error
            return await handler(request)

        return checked


router = APIRouter(prefix="/routing", route_class=RoutingJSONRoute,
                   responses={status: {"model": ErrorResponse} for status in (404, 409, 413, 422, 500, 503)})
Manager = Annotated[RoutingManager, Depends(get_routing_manager)]


@router.get("/{model_name}", response_model=RoutingResponse)
def get_routing(model_name: ModelName, manager: Manager) -> RoutingResponse:
    return RoutingResponse.from_snapshot(manager.get_routing_config(model_name))


@router.put("/{model_name}", response_model=RoutingResponse)
def configure(model_name: ModelName, body: RoutingUpdateRequest, manager: Manager) -> RoutingResponse:
    return RoutingResponse.from_snapshot(manager.configure_routing(model_name, body.weights, body.expected_revision))


@router.get("/{model_name}/history", response_model=list[RoutingResponse])
def history(model_name: ModelName, manager: Manager) -> list[RoutingResponse]:
    return [RoutingResponse.from_snapshot(item) for item in manager.get_routing_history(model_name)]


@router.post("/{model_name}/rollback", response_model=RoutingResponse)
def rollback(model_name: ModelName, body: RoutingRollbackRequest, manager: Manager) -> RoutingResponse:
    return RoutingResponse.from_snapshot(manager.rollback(model_name, body.target_revision, body.expected_revision))
