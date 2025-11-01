"""
Error handling and retry logic for SITV experiments.

This module provides decorators and utilities for handling common failures
in deep learning experiments, including CUDA OOM errors and evaluation failures.
"""

import time
import torch
import functools
from typing import Callable, Any, Optional, Tuple
import logging

# Set up logger
logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Exception raised when model evaluation fails."""

    pass


class CUDAOutOfMemoryError(Exception):
    """Exception raised when CUDA runs out of memory."""

    pass


def retry_on_cuda_oom(
    max_retries: int = 3, backoff_factor: float = 2.0, cleanup_fn: Optional[Callable] = None
):
    """Decorator to retry function on CUDA OOM errors.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier
        cleanup_fn: Optional cleanup function to call before retry

    Returns:
        Decorated function that retries on CUDA OOM

    Example:
        >>> @retry_on_cuda_oom(max_retries=3)
        ... def evaluate_model(model, inputs):
        ...     return model(**inputs)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        last_exception = CUDAOutOfMemoryError(f"CUDA OOM: {str(e)}")

                        if attempt < max_retries:
                            wait_time = backoff_factor**attempt
                            logger.warning(
                                f"CUDA OOM on attempt {attempt + 1}/{max_retries + 1}. "
                                f"Retrying in {wait_time:.1f}s..."
                            )

                            # Clean up CUDA memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()

                            # Call custom cleanup if provided
                            if cleanup_fn is not None:
                                cleanup_fn()

                            time.sleep(wait_time)
                        else:
                            logger.error(f"CUDA OOM after {max_retries + 1} attempts")
                            raise last_exception
                    else:
                        # Not a CUDA OOM error, re-raise immediately
                        raise

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def retry_on_evaluation_failure(
    max_retries: int = 2, backoff_factor: float = 1.5, return_on_failure: Any = float("inf")
):
    """Decorator to retry function on evaluation failures.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Exponential backoff multiplier
        return_on_failure: Value to return if all retries fail

    Returns:
        Decorated function that retries on failures

    Example:
        >>> @retry_on_evaluation_failure(max_retries=2, return_on_failure=float('inf'))
        ... def evaluate_loss(model, texts):
        ...     return compute_loss(model, texts)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_retries:
                        wait_time = backoff_factor**attempt
                        logger.warning(
                            f"Evaluation failed on attempt {attempt + 1}/{max_retries + 1}: {str(e)}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Evaluation failed after {max_retries + 1} attempts: {str(e)}. "
                            f"Returning fallback value: {return_on_failure}"
                        )
                        return return_on_failure

            # Should never reach here, but return fallback if it does
            return return_on_failure

        return wrapper

    return decorator


def handle_evaluation_error(
    error: Exception, alpha: float, context: str = "evaluation"
) -> Tuple[float, bool]:
    """Handle evaluation error with graceful degradation.

    Args:
        error: The exception that occurred
        alpha: Alpha value being evaluated
        context: Context string for logging

    Returns:
        Tuple of (fallback_loss, should_continue)
            - fallback_loss: Fallback loss value (inf)
            - should_continue: Whether experiment should continue

    Example:
        >>> try:
        ...     loss = evaluate(model, texts)
        ... except Exception as e:
        ...     loss, should_continue = handle_evaluation_error(e, alpha)
    """
    error_msg = str(error)

    # Check if it's a CUDA OOM error
    if "out of memory" in error_msg.lower():
        logger.error(
            f"CUDA OOM during {context} at α={alpha:.3f}. "
            f"Consider reducing batch size or model size."
        )
        # For OOM, we might want to stop the experiment
        return float("inf"), False

    # Check if it's a numerical instability
    if any(keyword in error_msg.lower() for keyword in ["nan", "inf", "numerical"]):
        logger.warning(
            f"Numerical instability during {context} at α={alpha:.3f}. "
            f"Continuing with fallback loss."
        )
        return float("inf"), True

    # Generic error - log and continue
    logger.warning(
        f"Error during {context} at α={alpha:.3f}: {error_msg}. Continuing with fallback loss."
    )
    return float("inf"), True


def safe_cuda_cleanup():
    """Safely clean up CUDA memory.

    Call this function to free up GPU memory when recovering from errors.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("CUDA memory cleaned up successfully")
        except Exception as e:
            logger.warning(f"Failed to clean up CUDA memory: {str(e)}")


class FailureTracker:
    """Track evaluation failures during experiments.

    This class helps identify patterns in failures and decide when
    to abort an experiment.

    Attributes:
        failures: List of (alpha, error_message) tuples
        max_consecutive_failures: Maximum allowed consecutive failures
        max_total_failures_pct: Maximum allowed failure percentage
    """

    def __init__(self, max_consecutive_failures: int = 5, max_total_failures_pct: float = 0.3):
        """Initialize failure tracker.

        Args:
            max_consecutive_failures: Max consecutive failures before abort
            max_total_failures_pct: Max failure percentage before abort (0-1)
        """
        self.failures: list[tuple[float, str]] = []
        self.consecutive_failures = 0
        self.total_evaluations = 0
        self.max_consecutive_failures = max_consecutive_failures
        self.max_total_failures_pct = max_total_failures_pct

    def record_success(self, alpha: float):
        """Record successful evaluation.

        Args:
            alpha: Alpha value that succeeded
        """
        self.consecutive_failures = 0
        self.total_evaluations += 1

    def record_failure(self, alpha: float, error: Exception):
        """Record failed evaluation.

        Args:
            alpha: Alpha value that failed
            error: Exception that occurred
        """
        self.failures.append((alpha, str(error)))
        self.consecutive_failures += 1
        self.total_evaluations += 1

    def should_abort(self) -> Tuple[bool, str]:
        """Check if experiment should be aborted.

        Returns:
            Tuple of (should_abort, reason)
        """
        # Check consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            return True, f"Too many consecutive failures ({self.consecutive_failures})"

        # Check total failure rate
        if self.total_evaluations > 0:
            failure_rate = len(self.failures) / self.total_evaluations
            if failure_rate > self.max_total_failures_pct:
                return True, f"Failure rate too high ({failure_rate:.1%})"

        return False, ""

    def get_summary(self) -> str:
        """Get failure summary string.

        Returns:
            Summary string with failure statistics
        """
        if self.total_evaluations == 0:
            return "No evaluations yet"

        failure_rate = len(self.failures) / self.total_evaluations
        return (
            f"Failures: {len(self.failures)}/{self.total_evaluations} "
            f"({failure_rate:.1%}), "
            f"Consecutive: {self.consecutive_failures}"
        )
