"""FastAPI application factory; module import opens no database or storage."""

from contextlib import asynccontextmanager
import logging
from threading import Lock

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from app.api.errors import install_exception_handlers
from app.api.middleware import RequestBodyLimitMiddleware
from app.api.routers.inference import router
from app.api.routers.routing import router as routing_router
from app.api.routers.monitoring import router as monitoring_router
from app.config import Settings
from app.core.container import ServiceContainer
from app.core.exceptions import StartupRecoveryError


def create_app(
    settings: Settings | None = None, *, container: ServiceContainer | None = None,
) -> FastAPI:
    if settings is not None and container is not None and settings != container.settings:
        raise ValueError("Injected container and application settings must match")
    selected = container.settings if container is not None else settings or Settings()
    lifecycle_lock = Lock()
    running = False

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal running
        with lifecycle_lock:
            if running:
                raise StartupRecoveryError("Application lifespan is already active")
            running = True
        active = None
        try:
            try:
                active = container if container is not None else await run_in_threadpool(ServiceContainer, selected)
                report = await run_in_threadpool(active.reconcile_startup)
            except Exception:
                logging.getLogger(__name__).error("application_startup_failed: infrastructure recovery was not completed")
                # Avoid framework startup traces printing backend paths/messages.
                raise StartupRecoveryError("Application startup failed; inspect sanitized recovery diagnostics") from None
            app.state.startup_report = report
            app.state.container = active
            yield
        finally:
            app.state.container = None
            app.state.startup_report = None
            try:
                if container is None and active is not None:
                    await run_in_threadpool(active.close)
            finally:
                with lifecycle_lock:
                    running = False

    app = FastAPI(title=selected.app_name, lifespan=lifespan)
    app.state.container = None
    app.state.startup_report = None
    app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=selected.max_request_body_bytes)
    install_exception_handlers(app)
    app.include_router(router)
    app.include_router(routing_router)
    app.include_router(monitoring_router)
    return app
