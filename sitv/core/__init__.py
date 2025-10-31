"""Core ML operations for SITV."""

from sitv.core.device import get_device_map, get_device, get_device_string
from sitv.core.task_vector import TaskVectorService
from sitv.core.evaluation import EvaluationService

__all__ = [
    "get_device_map",
    "get_device",
    "get_device_string",
    "TaskVectorService",
    "EvaluationService",
]
