"""Explicit or routed inference, guarded by deployment revision admission."""

from dataclasses import dataclass
import logging
from uuid import uuid4
from time import perf_counter_ns
from typing import Any

from app.core.exceptions import (
    DeploymentConflictError, DeploymentNotFoundError, DomainValidationError,
    InferenceUnavailableError, PredictionError, PredictionInputError,
)
from app.domain._validation import nonempty
from app.domain.inference_log import InferenceLog
from app.services.metrics_collector import MetricsCollector, classify_error
from app.domain.enums import DeploymentStatus, ModelVersionStatus
from app.interfaces.deployment_repository import DeploymentRepository
from app.interfaces.model_repository import ModelRepository
from app.services.deployment_health_checker import DeploymentHealthChecker
from app.services.inference_data import PredictionData
from app.services.model_runtime_manager import ModelRuntimeManager
from app.services.traffic_router import TrafficRouter


@dataclass(frozen=True)
class InferenceResult:
    model_name: str
    version: str
    predictions: Any
    latency_ms: float


class InferenceService:
    """One runtime retrieval and one vectorized prediction per request.

    Admission is the last successful fresh snapshot read after runtime readiness,
    with RUNNING and the original revision intact. A subsequent stop may complete
    while an admitted prediction runs. No transaction spans prediction, and no
    cross-process inference lease is claimed. All candidates refer to the exact
    resolved version. Automatic requests select once before the same admission
    path; explicit requests never call the router. No fallback is attempted.
    """

    def __init__(
        self, models: ModelRepository, deployments: DeploymentRepository,
        runtimes: ModelRuntimeManager, health_checker: DeploymentHealthChecker, *,
        max_batch_size: int = 128, max_tensor_dimensions: int = 4,
        max_tensor_elements: int = 100000,
        traffic_router: TrafficRouter | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        if any(type(value) is not int or value <= 0 for value in (
            max_batch_size, max_tensor_dimensions, max_tensor_elements,
        )):
            raise DomainValidationError("Inference limits must be positive integers")
        self._models = models
        self._deployments = deployments
        self._runtimes = runtimes
        self._health_checker = health_checker
        self._traffic_router = traffic_router
        self._metrics_collector = metrics_collector
        self._data = PredictionData(max_batch_size, max_tensor_dimensions, max_tensor_elements)

    def predict(self, model_name: str, version: str | None, inputs: Any) -> InferenceResult:
        started = perf_counter_ns()
        selected = version if isinstance(version, str) and version.strip() else None
        try:
            nonempty(model_name, "model_name")
            if not isinstance(inputs, (list, dict)) or not inputs:
                raise PredictionInputError("Inputs must be a nonempty tensor or tensor mapping")
            if version is None:
                if self._traffic_router is None:
                    raise InferenceUnavailableError("Automatic routing is not configured")
                selected = self._traffic_router.select(model_name).version
            nonempty(selected, "version")
            result = self._execute(model_name, selected, inputs, started)
        except Exception as error:
            self._observe(model_name, selected, (perf_counter_ns() - started) / 1_000_000, error)
            raise
        self._observe(model_name, selected, result.latency_ms, None)
        return result

    def _observe(self, model_name: str, version: str | None, latency_ms: float, error: Exception | None) -> None:
        if self._metrics_collector is None:
            return
        try:
            observation = InferenceLog(
                request_id=uuid4().hex,
                model_name=model_name if isinstance(model_name, str) and model_name.strip() else None,
                version=version, latency_ms=latency_ms, success=error is None,
                error_message=classify_error(error) if error is not None else None,
            )
            self._metrics_collector.record(observation)
        except Exception:
            # Never include payloads, backend messages, paths, or exception traces.
            logging.getLogger(__name__).error("monitoring_write_failed: observation persistence was not confirmed")

    def _execute(self, model_name: str, version: str, inputs: Any, started: int) -> InferenceResult:
        self._models.get_model(model_name)
        metadata = self._models.get_version(model_name, version)
        if metadata.status is not ModelVersionStatus.REGISTERED:
            raise InferenceUnavailableError("Version is not eligible for inference")
        self._data.validate_inputs(inputs, metadata.framework)
        self._data.validate_schema(inputs, metadata.input_schema)
        candidates = [snapshot for snapshot in self._deployments.list_deployments(model_name)
                      if snapshot.deployment.version == version
                      and snapshot.deployment.status is DeploymentStatus.RUNNING]
        if not candidates:
            raise InferenceUnavailableError("No RUNNING deployment for the requested version")
        runtime = self._runtimes.get_runtime(model_name, version)
        self._health_checker.check(runtime)  # No fabricated smoke input or duplicate prediction.
        changed = False
        for snapshot in candidates:
            try:
                current = self._deployments.get_deployment(snapshot.deployment.id)
            except DeploymentNotFoundError:
                continue
            deployment = current.deployment
            if (deployment.model_name != model_name or deployment.version != version
                    or deployment.status is not DeploymentStatus.RUNNING):
                continue
            if current.revision != snapshot.revision:
                changed = True
                continue
            break  # Admission point: this fresh RUNNING revision was observed.
        else:
            if changed:
                raise DeploymentConflictError("Deployment changed during inference admission")
            raise InferenceUnavailableError("Deployment stopped before inference admission")
        try:
            raw = runtime.predict(inputs)
        except PredictionError:
            raise
        except Exception as error:
            raise PredictionError("Loaded runtime prediction failed") from error
        predictions = self._data.normalize_outputs(raw)
        self._data.validate_schema(predictions, metadata.output_schema, output=True)
        return InferenceResult(model_name, version, predictions, (perf_counter_ns() - started) / 1_000_000)

    def batch_predict(self, model_name: str, version: str | None, inputs: Any) -> InferenceResult:
        """Execute one batched tensor call, not a list of independent requests.

        Both endpoints accept the same bounded batched inputs. This explicit alias
        preserves backend output ordering without repeated loading/execution.
        """
        return self.predict(model_name, version, inputs)
