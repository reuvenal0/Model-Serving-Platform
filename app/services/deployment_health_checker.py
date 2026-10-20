"""Conservative runtime readiness and optional model-specific smoke checks."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from app.core.exceptions import DeploymentHealthCheckError
from app.domain._validation import utc_now
from app.interfaces.loaded_model import LoadedModel


NO_SMOKE_INPUT = object()


@dataclass(frozen=True)
class ReadinessCheck:
    kind: Literal["load", "smoke"]
    checked_at: datetime = field(default_factory=utc_now)


class DeploymentHealthChecker:
    """Check an already loaded runtime, never deserialize artifacts itself.

    A load check proves runtime presence and the predict capability, not that
    arbitrary inputs work. A smoke check invokes predict with caller-provided,
    trusted model-specific input and proves only that this invocation completed.
    It does not validate accuracy or fabricate universal features.
    """

    def check(
        self, runtime: LoadedModel, *, smoke_inputs: Any = NO_SMOKE_INPUT,
    ) -> ReadinessCheck:
        if not callable(getattr(runtime, "predict", None)):
            raise DeploymentHealthCheckError("Loaded runtime is absent or has no predict()")
        if smoke_inputs is NO_SMOKE_INPUT:
            return ReadinessCheck("load")
        try:
            runtime.predict(smoke_inputs)
        except Exception as error:
            raise DeploymentHealthCheckError("Model-specific prediction smoke check failed") from error
        return ReadinessCheck("smoke")
