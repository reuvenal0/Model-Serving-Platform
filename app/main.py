"""FastAPI application factory; module import opens no database or storage."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.api.errors import install_exception_handlers
from app.api.middleware import RequestBodyLimitMiddleware
from app.api.routers.inference import router
from app.api.routers.routing import router as routing_router
from app.config import Settings
from app.core.container import ServiceContainer


def create_app(
    settings: Settings | None = None, *, container: ServiceContainer | None = None,
) -> FastAPI:
    if settings is not None and container is not None and settings != container.settings:
        raise ValueError("Injected container and application settings must match")
    selected = container.settings if container is not None else settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active = container if container is not None else await run_in_threadpool(ServiceContainer, selected)
        app.state.container = active
        try:
            yield
        finally:
            app.state.container = None
            if container is None:
                await run_in_threadpool(active.close)

    app = FastAPI(title=selected.app_name, lifespan=lifespan)
    app.state.container = None
    app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=selected.max_request_body_bytes)
    install_exception_handlers(app)
    app.include_router(router)
    app.include_router(routing_router)
    return app
