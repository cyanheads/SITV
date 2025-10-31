"""
Input validation for SITV experiments.

This module provides validators for experiment configurations and inputs,
ensuring experiments fail fast with clear error messages.
"""

from typing import Dict, List, Optional, Tuple
import torch


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


def validate_alpha_range(alpha_range: Tuple[float, float], name: str = "alpha_range") -> None:
    """Validate alpha range configuration.

    Args:
        alpha_range: Tuple of (min, max) alpha values
        name: Name of the range parameter (for error messages)

    Raises:
        ValidationError: If range is invalid
    """
    if not isinstance(alpha_range, (tuple, list)):
        raise ValidationError(
            f"{name} must be a tuple or list, got {type(alpha_range).__name__}"
        )

    if len(alpha_range) != 2:
        raise ValidationError(
            f"{name} must have exactly 2 elements (min, max), got {len(alpha_range)}"
        )

    min_val, max_val = alpha_range

    if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
        raise ValidationError(
            f"{name} values must be numeric, got ({type(min_val).__name__}, {type(max_val).__name__})"
        )

    if min_val >= max_val:
        raise ValidationError(
            f"{name} min must be less than max, got [{min_val}, {max_val}]"
        )


def validate_num_samples(num_samples: int, min_samples: int = 2) -> None:
    """Validate number of samples.

    Args:
        num_samples: Number of samples to generate
        min_samples: Minimum required samples

    Raises:
        ValidationError: If num_samples is invalid
    """
    if not isinstance(num_samples, int):
        raise ValidationError(
            f"num_samples must be an integer, got {type(num_samples).__name__}"
        )

    if num_samples < min_samples:
        raise ValidationError(
            f"num_samples must be at least {min_samples}, got {num_samples}"
        )


def validate_eval_texts(
    eval_texts: List[str],
    name: str = "eval_texts",
    allow_empty: bool = False
) -> None:
    """Validate evaluation text list.

    Args:
        eval_texts: List of evaluation texts
        name: Name of the texts parameter (for error messages)
        allow_empty: Whether to allow empty list

    Raises:
        ValidationError: If eval_texts is invalid
    """
    if not isinstance(eval_texts, list):
        raise ValidationError(
            f"{name} must be a list, got {type(eval_texts).__name__}"
        )

    if not allow_empty and len(eval_texts) == 0:
        raise ValidationError(
            f"{name} cannot be empty"
        )

    non_string_items = [i for i, text in enumerate(eval_texts) if not isinstance(text, str)]
    if non_string_items:
        raise ValidationError(
            f"{name} must contain only strings. "
            f"Non-string items at indices: {non_string_items[:5]}"
        )

    empty_items = [i for i, text in enumerate(eval_texts) if len(text.strip()) == 0]
    if empty_items:
        raise ValidationError(
            f"{name} contains empty strings at indices: {empty_items[:5]}"
        )


def validate_categories(
    categories: List[str],
    texts: List[str],
    name: str = "categories"
) -> None:
    """Validate category labels match texts.

    Args:
        categories: List of category labels
        texts: List of texts
        name: Name of the categories parameter

    Raises:
        ValidationError: If categories don't match texts
    """
    if not isinstance(categories, list):
        raise ValidationError(
            f"{name} must be a list, got {type(categories).__name__}"
        )

    if len(categories) != len(texts):
        raise ValidationError(
            f"{name} length ({len(categories)}) must match texts length ({len(texts)})"
        )

    non_string_items = [i for i, cat in enumerate(categories) if not isinstance(cat, str)]
    if non_string_items:
        raise ValidationError(
            f"{name} must contain only strings. "
            f"Non-string items at indices: {non_string_items[:5]}"
        )


def validate_task_vector(task_vector: Dict[str, torch.Tensor], name: str = "task_vector") -> None:
    """Validate task vector dictionary.

    Args:
        task_vector: Task vector dictionary (param_name -> tensor)
        name: Name of the task vector parameter

    Raises:
        ValidationError: If task vector is invalid
    """
    if not isinstance(task_vector, dict):
        raise ValidationError(
            f"{name} must be a dictionary, got {type(task_vector).__name__}"
        )

    if len(task_vector) == 0:
        raise ValidationError(
            f"{name} cannot be empty"
        )

    non_tensor_keys = [
        key for key, value in task_vector.items()
        if not isinstance(value, torch.Tensor)
    ]
    if non_tensor_keys:
        raise ValidationError(
            f"{name} must contain only torch.Tensor values. "
            f"Non-tensor keys: {non_tensor_keys[:5]}"
        )

    # Check for NaN or Inf values
    invalid_keys = []
    for key, tensor in task_vector.items():
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            invalid_keys.append(key)

    if invalid_keys:
        raise ValidationError(
            f"{name} contains NaN or Inf values in tensors: {invalid_keys[:5]}"
        )


def validate_alpha_sweep_config(
    alpha_range: Tuple[float, float],
    num_samples: int,
    general_eval_texts: List[str],
    task_eval_texts: List[str],
    task_vector: Dict[str, torch.Tensor],
    general_eval_categories: Optional[List[str]] = None,
) -> None:
    """Validate complete alpha sweep configuration.

    This is a convenience function that runs all relevant validators.

    Args:
        alpha_range: Range of alpha values (min, max)
        num_samples: Number of alpha samples
        general_eval_texts: General evaluation texts
        task_eval_texts: Task-specific evaluation texts
        task_vector: Task vector dictionary
        general_eval_categories: Optional category labels for general texts

    Raises:
        ValidationError: If any validation fails
    """
    validate_alpha_range(alpha_range, "alpha_range")
    validate_num_samples(num_samples, min_samples=2)
    validate_eval_texts(general_eval_texts, "general_eval_texts")
    validate_eval_texts(task_eval_texts, "task_eval_texts")
    validate_task_vector(task_vector, "task_vector")

    if general_eval_categories is not None:
        validate_categories(general_eval_categories, general_eval_texts, "general_eval_categories")


def validate_2d_composition_config(
    alpha_range: Tuple[float, float],
    beta_range: Tuple[float, float],
    num_samples_per_dim: int,
    general_eval_texts: List[str],
    task_vector_1: Dict[str, torch.Tensor],
    task_vector_2: Dict[str, torch.Tensor],
) -> None:
    """Validate 2D composition experiment configuration.

    Args:
        alpha_range: Range of alpha values (min, max)
        beta_range: Range of beta values (min, max)
        num_samples_per_dim: Number of samples per dimension
        general_eval_texts: General evaluation texts
        task_vector_1: First task vector
        task_vector_2: Second task vector

    Raises:
        ValidationError: If any validation fails
    """
    validate_alpha_range(alpha_range, "alpha_range")
    validate_alpha_range(beta_range, "beta_range")
    validate_num_samples(num_samples_per_dim, min_samples=2)
    validate_eval_texts(general_eval_texts, "general_eval_texts")
    validate_task_vector(task_vector_1, "task_vector_1")
    validate_task_vector(task_vector_2, "task_vector_2")
