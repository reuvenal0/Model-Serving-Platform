"""FastAPI application factory; module import opens no database or storage."""

from contextlib import asynccontextmanager
import logging
from threading import Lock

from fastapi import FastAPI
from fastapi import Depends
from fastapi.openapi.docs import (
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.errors import install_exception_handlers
from app.api.middleware import RequestBodyLimitMiddleware, SanitizedErrorMiddleware
from app.api.routers.inference import router
from app.api.routers.routing import router as routing_router
from app.api.routers.monitoring import router as monitoring_router
from app.api.routers.management import router as management_router
from app.config import Settings
from app.core.container import ServiceContainer
from app.core.exceptions import StartupRecoveryError
from app.api.security import require_documentation_access
from pathlib import Path
from app.services.inference_admission import InferenceAdmission


def create_app(
    settings: Settings | None = None,
    *,
    container: ServiceContainer | None = None,
) -> FastAPI:
    if (
        settings is not None
        and container is not None
        and settings != container.settings
    ):
        raise ValueError("Injected container and application settings must match")
    selected = container.settings if container is not None else settings or Settings()
    if selected.enable_dev_swagger and Path("/.dockerenv").exists():
        raise ValueError(
            "Development Swagger is unsupported inside Docker; use the container-local management channel"
        )
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
        admission = None
        try:
            try:
                active = (
                    container
                    if container is not None
                    else await run_in_threadpool(ServiceContainer, selected)
                )
                report = await run_in_threadpool(active.reconcile_startup)
            except Exception:
                logging.getLogger(__name__).error(
                    "application_startup_failed: infrastructure recovery was not completed"
                )
                # Avoid framework startup traces printing backend paths/messages.
                raise StartupRecoveryError(
                    "Application startup failed; inspect sanitized recovery diagnostics"
                ) from None
            app.state.startup_report = report
            app.state.container = active
            admission = InferenceAdmission(
                selected.max_concurrent_inference,
                selected.max_waiting_inference,
                selected.admission_wait_timeout_seconds,
            )
            app.state.inference_admission = admission
            yield
        finally:
            try:
                if admission is not None:
                    await admission.close()
            finally:
                app.state.inference_admission = None
                app.state.container = None
                app.state.startup_report = None
                try:
                    if container is None and active is not None:
                        await run_in_threadpool(active.close)
                finally:
                    with lifecycle_lock:
                        running = False

    app = FastAPI(
        title=selected.app_name,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.security_settings = selected
    app.state.container = None
    app.state.startup_report = None
    app.add_middleware(
        RequestBodyLimitMiddleware, max_body_bytes=selected.max_request_body_bytes
    )
    app.add_middleware(SanitizedErrorMiddleware)
    install_exception_handlers(app)
    app.include_router(router)
    app.include_router(routing_router)
    app.include_router(monitoring_router)
    app.include_router(management_router)

    @app.get(
        "/openapi.json",
        dependencies=[Depends(require_documentation_access)],
        include_in_schema=False,
    )
    async def schema():
        return JSONResponse(app.openapi())

    @app.get(
        "/docs",
        dependencies=[Depends(require_documentation_access)],
        include_in_schema=False,
    )
    async def docs():
        # Inline the schema so Swagger need not send the viewing credential again.
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=selected.app_name,
            oauth2_redirect_url="/docs/oauth2-redirect",
            swagger_ui_parameters={"spec": app.openapi()},
        )

    @app.get(
        "/docs/oauth2-redirect",
        dependencies=[Depends(require_documentation_access)],
        include_in_schema=False,
    )
    async def docs_redirect():
        return get_swagger_ui_oauth2_redirect_html()

    return app
