"""Supported frameworks and independent lifecycle states."""

from enum import Enum


class ModelFramework(str, Enum):
    SKLEARN = "sklearn"
    ONNX = "onnx"


class ModelVersionStatus(str, Enum):
    REGISTERED = "registered"
    INVALID = "invalid"


class DeploymentStatus(str, Enum):
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"
