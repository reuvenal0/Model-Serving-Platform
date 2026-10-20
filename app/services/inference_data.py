"""Narrow tensor validation and bounded JSON prediction conversion."""

from math import isfinite
from typing import Any

import numpy as np

from app.core.exceptions import ModelContractError, PredictionInputError, PredictionSerializationError
from app.domain.enums import ModelFramework


class PredictionData:
    def __init__(self, max_batch_size: int, max_dimensions: int, max_elements: int) -> None:
        self._max_batch_size = max_batch_size
        self._max_dimensions = max_dimensions
        self._max_elements = max_elements

    def _tensor(self, value: Any, budget: list[int], *, allow_empty: bool = False) -> tuple[tuple[int, ...], str]:
        kinds = set()

        def visit(item, depth):
            if isinstance(item, list):
                if depth >= self._max_dimensions:
                    raise PredictionInputError("Tensor rank exceeds configured limit")
                if not item:
                    if allow_empty:
                        return (0,)
                    raise PredictionInputError("Empty tensors and batches are not supported")
                shape = visit(item[0], depth + 1)
                for child in item[1:]:
                    if visit(child, depth + 1) != shape:
                        raise PredictionInputError("Tensor must be rectangular")
                return (len(item),) + shape
            budget[0] += 1
            if budget[0] > self._max_elements:
                raise PredictionInputError("Tensor element limit exceeded")
            if type(item) is bool:
                kinds.add("boolean")
            elif type(item) in (int, float):
                try:
                    finite = isfinite(item)
                except OverflowError:
                    finite = False
                if not finite:
                    raise PredictionInputError("Tensor numbers must be finite and representable")
                kinds.add("integer" if type(item) is int else "number")
            elif type(item) is str:
                kinds.add("string")
            else:
                raise PredictionInputError("Unsupported tensor scalar type")
            return ()

        shape = visit(value, 0)
        if kinds <= {"integer", "number"}:
            kind = "number" if "number" in kinds else "integer"
        elif len(kinds) == 1:
            kind = next(iter(kinds))
        else:
            raise PredictionInputError("Tensor scalar types cannot be mixed")
        return shape, kind

    def validate_inputs(self, inputs: Any, framework: ModelFramework) -> None:
        if isinstance(inputs, dict):
            if framework is not ModelFramework.ONNX or not inputs:
                raise PredictionInputError("Named tensors require a nonempty ONNX input mapping")
            if any(not isinstance(name, str) or not name.strip() for name in inputs):
                raise PredictionInputError("Tensor names must be nonempty strings")
            tensors = list(inputs.values())
        elif isinstance(inputs, list):
            tensors = [inputs]
        else:
            raise PredictionInputError("Inputs must be a tensor list or named ONNX tensor mapping")
        budget = [0]
        batch_size = None
        for tensor in tensors:
            if not isinstance(tensor, list):
                raise PredictionInputError("Each tensor must be a list")
            shape, kind = self._tensor(tensor, budget)
            if not shape or not 1 <= shape[0] <= self._max_batch_size:
                raise PredictionInputError("Batch size exceeds configured bounds")
            if batch_size is not None and batch_size != shape[0]:
                raise PredictionInputError("Named tensors must have matching batch dimensions")
            batch_size = shape[0]
            if kind == "string" or (framework is ModelFramework.SKLEARN and (len(shape) != 2 or kind == "boolean")):
                raise PredictionInputError("sklearn requires numeric matrices; ONNX supports numeric or boolean tensors")

    def normalize_outputs(self, value: Any) -> Any:
        budget = [0]

        def normalize(item, depth):
            if depth > self._max_dimensions + 4:
                raise PredictionSerializationError("Prediction nesting exceeds configured bounds")
            if isinstance(item, np.ndarray):
                if item.size > self._max_elements or item.ndim > self._max_dimensions:
                    raise PredictionSerializationError("Prediction tensor exceeds configured bounds")
                return normalize(item.tolist(), depth)
            if isinstance(item, np.generic):
                converted = item.item()
                if isinstance(converted, np.generic):
                    raise PredictionSerializationError("Unsupported extended NumPy scalar")
                return normalize(converted, depth)
            if isinstance(item, dict):
                if len(item) > self._max_elements or any(type(key) is not str for key in item):
                    raise PredictionSerializationError("Prediction object keys must be strings")
                return {key: normalize(child, depth + 1) for key, child in item.items()}
            if isinstance(item, (list, tuple)):
                if len(item) > self._max_elements:
                    raise PredictionSerializationError("Prediction element limit exceeded")
                return [normalize(child, depth + 1) for child in item]
            budget[0] += 1
            if budget[0] > self._max_elements:
                raise PredictionSerializationError("Prediction element limit exceeded")
            if item is None or type(item) in (bool, str, int):
                return item
            if type(item) is float and isfinite(item):
                return item
            raise PredictionSerializationError("Prediction contains an unsupported JSON value")

        return normalize(value, 0)

    def validate_schema(self, value: Any, schema: dict, *, output: bool = False) -> None:
        """Validate only an explicit 'inference' shape/dtype contract.

        Existing metadata outside this key is opaque, not arbitrary JSON Schema.
        Contracts are {shape: [int|null, ...], dtype: number|integer|boolean|string}
        or {tensors: {name: contract, ...}} for exact named tensor mappings.
        """
        if not isinstance(schema, dict):
            raise ModelContractError("Schema metadata must be a dictionary")
        if "inference" not in schema:
            return
        contract = schema["inference"]
        if not isinstance(contract, dict):
            raise ModelContractError("inference contract must be a dictionary")
        if set(contract) == {"tensors"}:
            specs = contract["tensors"]
            if not isinstance(specs, dict) or not specs or any(type(name) is not str for name in specs):
                raise ModelContractError("Named inference contracts must contain tensor specifications")
        else:
            specs = {None: contract}
        for spec in specs.values():
            if not isinstance(spec, dict) or set(spec) != {"shape", "dtype"}:
                raise ModelContractError("Only shape and dtype tensor constraints are supported")
            shape = spec["shape"]
            if (not isinstance(shape, list) or len(shape) > self._max_dimensions
                    or any(item is not None and (type(item) is not int or item < 0) for item in shape)
                    or spec["dtype"] not in ("number", "integer", "boolean", "string")):
                raise ModelContractError("Unsupported inference shape/dtype specification")
        mismatch = PredictionSerializationError if output else PredictionInputError
        if None not in specs:
            if not isinstance(value, dict) or set(value) != set(specs):
                raise mismatch("Tensor names do not match the inference contract")
        for name, spec in specs.items():
            tensor = value if name is None else value[name]
            try:
                shape, kind = self._tensor(tensor, [0], allow_empty=output)
            except PredictionInputError as error:
                raise mismatch("Data does not satisfy the inference tensor contract") from error
            if (len(shape) != len(spec["shape"])
                    or any(expected is not None and expected != actual for actual, expected in zip(shape, spec["shape"]))
                    or not (kind == spec["dtype"] or (spec["dtype"] == "number" and kind == "integer"))):
                raise mismatch("Data does not satisfy the inference tensor contract")
