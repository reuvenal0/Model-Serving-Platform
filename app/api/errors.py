"""Sanitized, stable HTTP error responses without backend exception details."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from app.core import exceptions as errors


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


_MAPPINGS = (
    ((errors.PredictionInputError, errors.DomainValidationError), 422, "invalid_input", "Prediction input is invalid"),
    ((errors.ModelNotFoundError, errors.ModelVersionNotFoundError, errors.DeploymentNotFoundError),
     404, "not_found", "Requested model, version, or deployment was not found"),
    ((errors.DeploymentConflictError, errors.InvalidDeploymentStateError, errors.ModelVersionInUseError,
      errors.DuplicateModelError, errors.DuplicateVersionError, errors.DuplicateDeploymentError),
     409, "lifecycle_conflict", "Deployment lifecycle changed or the operation conflicts with current state"),
    ((errors.InferenceUnavailableError, errors.ModelLoadingError, errors.ModelContractError,
      errors.DeploymentHealthCheckError, errors.DeploymentRecoveryError, errors.RepositoryError,
      errors.ArtifactStorageError, errors.UnsupportedFrameworkError),
     503, "model_unavailable", "Model or supporting service is unavailable"),
    ((errors.PredictionError,), 500, "prediction_failed", "Prediction could not be completed"),
)


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        # Pydantic's raw errors can include user inputs and exception contexts.
        return error_response(422, "invalid_request", "Request body or parameters are invalid")

    @app.exception_handler(errors.DomainError)
    async def domain_error(request: Request, error: errors.DomainError):
        for classes, status, code, message in _MAPPINGS:
            if isinstance(error, classes):
                return error_response(status, code, message)
        return error_response(500, "internal_error", "Request could not be completed")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException):
        return error_response(error.status_code, "http_error", "HTTP request could not be completed")

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception):
        logging.getLogger(__name__).error("Unhandled API error", exc_info=error)
        return error_response(500, "internal_error", "Request could not be completed")
