from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.core.exceptions import (
    DeploymentConflictError, DomainValidationError, InferenceUnavailableError,
    ModelContractError, ModelLoadingError, ModelVersionNotFoundError,
    PredictionError, PredictionInputError, PredictionSerializationError,
)
from app.domain.deployment import Deployment
from app.domain.enums import DeploymentStatus, ModelFramework, ModelVersionStatus
from app.domain.model import Model
from app.domain.model_version import ModelVersion
from app.domain.resource_config import ResourceConfig
from app.interfaces.deployment_repository import DeploymentRepository, DeploymentSnapshot
from app.interfaces.model_repository import ModelRepository
from app.services.deployment_health_checker import DeploymentHealthChecker
from app.services.inference_service import InferenceService
from app.services.model_runtime_manager import ModelRuntimeManager


@pytest.fixture
def system():
    models = Mock(spec=ModelRepository)
    models.get_model.return_value = Model("model")
    metadata = ModelVersion("model", "v1", "key", ModelFramework.SKLEARN)
    models.get_version.return_value = metadata
    deployment = Deployment("d-1", "model", "v1", 1, ResourceConfig(1, 128), DeploymentStatus.RUNNING)
    snapshot = DeploymentSnapshot(deployment, 1)
    deployments = Mock(spec=DeploymentRepository)
    deployments.list_deployments.return_value = [snapshot]
    deployments.get_deployment.side_effect = lambda identifier: DeploymentSnapshot(replace(deployment), 1)
    runtime = Mock()
    runtime.predict.return_value = np.array([7.0, 9.0])
    runtimes = Mock(spec=ModelRuntimeManager)
    runtimes.get_runtime.return_value = runtime
    checker = DeploymentHealthChecker()
    service = InferenceService(models, deployments, runtimes, checker, max_batch_size=4)
    return SimpleNamespace(models=models, metadata=metadata, deployments=deployments,
                           snapshot=snapshot, runtime=runtime, runtimes=runtimes,
                           checker=checker, service=service)


def test_vectorized_prediction_has_ordered_json_output_and_latency(system):
    result = system.service.predict("model", "v1", [[3.0], [4.0]])
    assert (result.model_name, result.version) == ("model", "v1")
    assert result.predictions == [7.0, 9.0]
    assert result.latency_ms >= 0
    system.runtimes.get_runtime.assert_called_once_with("model", "v1")
    system.runtime.predict.assert_called_once_with([[3.0], [4.0]])
    system.deployments.update_deployment.assert_not_called()


def test_batch_alias_executes_one_prediction_not_one_call_per_sample(system):
    result = system.service.batch_predict("model", "v1", [[4], [3]])
    assert result.predictions == [7.0, 9.0]
    system.runtime.predict.assert_called_once_with([[4], [3]])
    system.runtimes.get_runtime.assert_called_once()


@pytest.mark.parametrize("status", [DeploymentStatus.STOPPED, DeploymentStatus.FAILED, DeploymentStatus.DEPLOYING])
def test_nonrunning_versions_are_not_admitted(system, status):
    system.deployments.list_deployments.return_value = [DeploymentSnapshot(replace(system.snapshot.deployment, status=status), 2)]
    with pytest.raises(InferenceUnavailableError):
        system.service.predict("model", "v1", [[3]])
    system.runtimes.get_runtime.assert_not_called()


def test_other_version_running_does_not_qualify_requested_version(system):
    system.deployments.list_deployments.return_value = [DeploymentSnapshot(replace(system.snapshot.deployment, version="v2"), 1)]
    with pytest.raises(InferenceUnavailableError):
        system.service.predict("model", "v1", [[3]])
    system.runtime.predict.assert_not_called()


def test_unregistered_version_is_not_loaded(system):
    error = ModelVersionNotFoundError("missing")
    system.models.get_version.side_effect = error
    with pytest.raises(ModelVersionNotFoundError) as captured:
        system.service.predict("model", "missing", [[3]])
    assert captured.value is error
    system.runtimes.get_runtime.assert_not_called()


def test_invalid_registered_status_is_not_admitted(system):
    system.metadata.status = ModelVersionStatus.INVALID
    with pytest.raises(InferenceUnavailableError):
        system.service.predict("model", "v1", [[3]])
    system.runtime.predict.assert_not_called()


@pytest.mark.parametrize("inputs", [None, [], {}, [1, 2], [[1], [2, 3]], [[True]],
                                         [["1"]], [[float("nan")]], [[float("inf")]],
                                         [[1]] * 5, [[[[[1]]]]], [{"request": [[1]]}]])
def test_invalid_inputs_fail_before_loading(system, inputs):
    with pytest.raises(PredictionInputError):
        system.service.predict("model", "v1", inputs)
    system.runtimes.get_runtime.assert_not_called()


@pytest.mark.parametrize("name,version", [(" ", "v1"), ("model", "")])
def test_empty_identifiers(system, name, version):
    with pytest.raises(DomainValidationError):
        system.service.predict(name, version, [[1]])
    system.models.get_model.assert_not_called()


def test_named_onnx_tensors_are_passed_unchanged(system):
    system.metadata.framework = ModelFramework.ONNX
    inputs = {"X": [[1, 2]], "flags": [[True, False]]}
    system.runtime.predict.return_value = {"Y": np.array([[3, 5]], dtype=np.float32)}
    result = system.service.predict("model", "v1", inputs)
    assert result.predictions == {"Y": [[3.0, 5.0]]}
    system.runtime.predict.assert_called_once_with(inputs)


def test_mismatched_named_batch_dimensions_are_rejected(system):
    system.metadata.framework = ModelFramework.ONNX
    with pytest.raises(PredictionInputError):
        system.service.predict("model", "v1", {"X": [[1, 2]], "Z": [[1, 2], [3, 4]]})


@pytest.mark.parametrize("error", [ModelLoadingError("cannot load"), PredictionError("prediction failed"), PredictionInputError("bad shape")])
def test_runtime_and_prediction_errors_remain_explicit(system, error):
    if isinstance(error, ModelLoadingError):
        system.runtimes.get_runtime.side_effect = error
    else:
        system.runtime.predict.side_effect = error
    with pytest.raises(type(error)) as captured:
        system.service.predict("model", "v1", [[3]])
    assert captured.value is error
    system.deployments.update_deployment.assert_not_called()


def test_unexpected_prediction_error_is_chained(system):
    system.runtime.predict.side_effect = RuntimeError("backend broke")
    with pytest.raises(PredictionError) as captured:
        system.service.predict("model", "v1", [[3]])
    assert isinstance(captured.value.__cause__, RuntimeError)


@pytest.mark.parametrize("output", [object(), {1: "bad key"}, {1, 2}, complex(1, 2), b"bytes",
                                         float("nan"), np.array([float("inf")]), np.array([complex(1, 2)])])
def test_unsupported_outputs_fail_explicitly(system, output):
    system.runtime.predict.return_value = output
    with pytest.raises(PredictionSerializationError):
        system.service.predict("model", "v1", [[3]])


def test_nested_numpy_outputs_are_normalized_without_stringifying_values(system):
    system.runtime.predict.return_value = {
        "labels": np.array(["ham", "spam"]), "count": np.int64(2),
        "flag": np.bool_(True), "score": np.float32(0.5), "optional": None,
    }
    assert system.service.predict("model", "v1", [[3]]).predictions == {
        "labels": ["ham", "spam"], "count": 2, "flag": True, "score": 0.5, "optional": None,
    }


def test_opt_in_input_output_contracts_and_opaque_legacy_metadata(system):
    system.metadata.input_schema = {"type": "not-interpreted", "inference": {"shape": [None, 1], "dtype": "number"}}
    system.metadata.output_schema = {"inference": {"shape": [2], "dtype": "number"}}
    assert system.service.predict("model", "v1", [[3], [4]]).predictions == [7.0, 9.0]
    with pytest.raises(PredictionInputError):
        system.service.predict("model", "v1", [[3, 4]])
    system.metadata.output_schema["inference"]["shape"] = [3]
    with pytest.raises(PredictionSerializationError):
        system.service.predict("model", "v1", [[3], [4]])


@pytest.mark.parametrize("contract", [None, {}, {"shape": [None]}, {"shape": [True], "dtype": "number"}, {"shape": [1], "dtype": "complex"}, {"tensors": {}}])
def test_invalid_model_contract_is_not_misreported_as_client_input(system, contract):
    system.metadata.input_schema = {"inference": contract}
    with pytest.raises(ModelContractError):
        system.service.predict("model", "v1", [[3]])


def test_named_tensor_contracts(system):
    system.metadata.framework = ModelFramework.ONNX
    system.metadata.input_schema = {"inference": {"tensors": {"X": {"shape": [None, 2], "dtype": "integer"}}}}
    system.metadata.output_schema = {"inference": {"tensors": {"Y": {"shape": [None, 2], "dtype": "number"}}}}
    system.runtime.predict.return_value = {"Y": np.array([[3, 5]], dtype=np.float32)}
    assert system.service.predict("model", "v1", {"X": [[1, 2]]}).predictions == {"Y": [[3.0, 5.0]]}
    with pytest.raises(PredictionInputError):
        system.service.predict("model", "v1", {"X": [[1.5, 2.5]]})


def test_stopped_after_readiness_before_admission_never_predicts(system):
    stopped = replace(system.snapshot.deployment)
    stopped.stop()
    system.deployments.get_deployment.side_effect = lambda identifier: DeploymentSnapshot(stopped, 2)
    with pytest.raises(InferenceUnavailableError):
        system.service.predict("model", "v1", [[3]])
    system.runtime.predict.assert_not_called()


def test_scaling_revision_change_requires_new_admission(system):
    changed = replace(system.snapshot.deployment)
    changed.scale(3)
    system.deployments.get_deployment.side_effect = lambda identifier: DeploymentSnapshot(changed, 2)
    with pytest.raises(DeploymentConflictError):
        system.service.predict("model", "v1", [[3]])
    system.runtime.predict.assert_not_called()


def test_other_stable_running_deployment_can_admit_same_version(system):
    other = replace(system.snapshot.deployment, id="d-2")
    stopped = replace(system.snapshot.deployment)
    stopped.stop()
    system.deployments.list_deployments.return_value = [system.snapshot, DeploymentSnapshot(other, 1)]
    system.deployments.get_deployment.side_effect = [DeploymentSnapshot(stopped, 2), DeploymentSnapshot(other, 1)]
    assert system.service.predict("model", "v1", [[3]]).predictions == [7.0, 9.0]


def test_admitted_prediction_may_finish_after_stop(system):
    entered, release = Event(), Event()

    def predict(inputs):
        entered.set()
        assert release.wait(5)
        return np.array([7])

    system.runtime.predict.side_effect = predict
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(system.service.predict, "model", "v1", [[3]])
        assert entered.wait(5)
        try:
            stopped = replace(system.snapshot.deployment)
            stopped.stop()
            system.deployments.list_deployments.return_value = [DeploymentSnapshot(stopped, 2)]
            with pytest.raises(InferenceUnavailableError):
                system.service.predict("model", "v1", [[3]])
        finally:
            release.set()
        assert pending.result(timeout=5).predictions == [7]


def test_input_and_output_element_budgets_are_enforced(system):
    limited = InferenceService(system.models, system.deployments, system.runtimes,
                               system.checker, max_tensor_elements=2)
    with pytest.raises(PredictionInputError):
        limited.predict("model", "v1", [[1], [2], [3]])
    system.runtimes.get_runtime.assert_not_called()
    system.runtime.predict.return_value = np.array([1, 2, 3])
    with pytest.raises(PredictionSerializationError):
        limited.predict("model", "v1", [[1]])


@pytest.mark.parametrize("limit,value", [("max_batch_size", 0), ("max_tensor_dimensions", True),
                                         ("max_tensor_elements", -1)])
def test_inference_service_rejects_invalid_limits(system, limit, value):
    with pytest.raises(DomainValidationError):
        InferenceService(system.models, system.deployments, system.runtimes,
                         system.checker, **{limit: value})
