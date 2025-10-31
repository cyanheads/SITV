"""Core ML operations for SITV."""

from sitv.core.device import get_device_map, get_device, get_device_string
from sitv.core.task_vector import TaskVectorService
from sitv.core.evaluation import EvaluationService
from sitv.core.validation import (
    validate_alpha_range,
    validate_num_samples,
    validate_eval_texts,
    validate_task_vector,
    validate_alpha_sweep_config,
    validate_2d_composition_config,
    ValidationError,
)
from sitv.core.error_handling import (
    retry_on_cuda_oom,
    retry_on_evaluation_failure,
    handle_evaluation_error,
    safe_cuda_cleanup,
    FailureTracker,
    EvaluationError,
    CUDAOutOfMemoryError,
)

__all__ = [
    "get_device_map",
    "get_device",
    "get_device_string",
    "TaskVectorService",
    "EvaluationService",
    # Validation
    "validate_alpha_range",
    "validate_num_samples",
    "validate_eval_texts",
    "validate_task_vector",
    "validate_alpha_sweep_config",
    "validate_2d_composition_config",
    "ValidationError",
    # Error handling
    "retry_on_cuda_oom",
    "retry_on_evaluation_failure",
    "handle_evaluation_error",
    "safe_cuda_cleanup",
    "FailureTracker",
    "EvaluationError",
    "CUDAOutOfMemoryError",
]
