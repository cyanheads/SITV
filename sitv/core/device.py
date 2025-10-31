"""
Device management utilities for SITV.

This module provides utilities for detecting and managing compute devices
(CUDA, MPS, CPU) for model loading and inference.
"""

import torch


def get_device_map():
    """Get appropriate device_map for model loading based on available hardware.

    Returns:
        str or None: Device map configuration for transformers models.
        - "auto" for CUDA (multi-GPU support)
        - None for MPS or CPU (load to default device)

    Examples:
        >>> device_map = get_device_map()
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "model_name",
        ...     device_map=device_map
        ... )
    """
    if torch.cuda.is_available():
        return "auto"  # CUDA supports automatic device mapping
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return None  # MPS: load to default device, then move manually
    else:
        return None  # CPU: load to default device


def get_device() -> torch.device:
    """Get the primary compute device for model operations.

    Returns:
        torch.device: The device to use for computations.
        Priority: CUDA > MPS > CPU

    Examples:
        >>> device = get_device()
        >>> tensor = tensor.to(device)
        >>> model = model.to(device)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_string() -> str:
    """Get the device name as a string.

    Returns:
        str: Device name ("cuda", "mps", or "cpu")

    Examples:
        >>> device_str = get_device_string()
        >>> print(f"Using device: {device_str}")
    """
    return str(get_device())
