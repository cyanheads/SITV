"""
Base experiment class for SITV experiments.

This module provides the abstract base class for all SITV experiments.
"""

import time
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional
from transformers import PreTrainedModel


class Experiment(ABC):
    """Abstract base class for SITV experiments.

    All experiments should inherit from this class and implement the run() method.
    This class provides common functionality for:
    - Parameter cloning and restoration
    - Timing and progress tracking
    - Result collection

    Attributes:
        base_model: Base (pretrained) model
        tokenizer: HuggingFace tokenizer
        device: Device for computation
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer,
        device: str = "cuda"
    ):
        """Initialize the experiment.

        Args:
            base_model: Base model for the experiment
            tokenizer: Tokenizer for evaluation
            device: Device for computation
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    @abstractmethod
    def run(self) -> tuple[List[Any], Dict[str, Any]]:
        """Run the experiment.

        Returns:
            Tuple of (results_list, metadata_dict)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement run()")

    def clone_parameters(
        self,
        task_vector: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Clone model parameters for restoration later.

        Args:
            task_vector: Task vector to determine which parameters to clone

        Returns:
            Dictionary of cloned parameters
        """
        print("Cloning base model parameters...")
        original_params = {}
        for name, param in self.base_model.named_parameters():
            if name in task_vector:
                original_params[name] = param.clone().detach()
        return original_params

    def restore_parameters(
        self,
        original_params: Dict[str, torch.Tensor]
    ) -> None:
        """Restore model parameters to original state.

        Args:
            original_params: Dictionary of original parameters
        """
        print("\nRestoring base model parameters...")
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in original_params:
                    param.copy_(original_params[name])

    def preload_task_vector_to_device(
        self,
        task_vector: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Pre-move task vector tensors to model device.

        This avoids repeated device transfers in tight loops, significantly
        improving performance for large models.

        Args:
            task_vector: Task vector with tensors on any device

        Returns:
            Task vector with all tensors on the model's device
        """
        print("Pre-loading task vector to device...")
        device_task_vector = {}
        for name, tensor in task_vector.items():
            device_task_vector[name] = tensor.to(self.device)
        return device_task_vector

    def apply_task_vector(
        self,
        original_params: Dict[str, torch.Tensor],
        task_vector: Dict[str, torch.Tensor],
        alpha: float
    ) -> None:
        """Apply scaled task vector to model: M(α) = M_base + α·T

        Args:
            original_params: Original model parameters
            task_vector: Task vector to apply (should be pre-loaded to device)
            alpha: Scaling factor

        Note:
            For best performance, pre-load task_vector to device using
            preload_task_vector_to_device() before calling this in a loop.
        """
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in task_vector and name in original_params:
                    # Task vector should already be on correct device
                    param.copy_(
                        original_params[name] + alpha * task_vector[name]
                    )

    def apply_2d_composition(
        self,
        original_params: Dict[str, torch.Tensor],
        task_vector_1: Dict[str, torch.Tensor],
        task_vector_2: Dict[str, torch.Tensor],
        alpha: float,
        beta: float
    ) -> None:
        """Apply 2D task vector composition: M(α,β) = M_base + α·T1 + β·T2

        Args:
            original_params: Original model parameters
            task_vector_1: First task vector (should be pre-loaded to device)
            task_vector_2: Second task vector (should be pre-loaded to device)
            alpha: Scaling factor for T1
            beta: Scaling factor for T2

        Note:
            For best performance, pre-load both task vectors to device using
            preload_task_vector_to_device() before calling this in a loop.
        """
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if name in original_params:
                    # Task vectors should already be on correct device
                    param.copy_(
                        original_params[name]
                        + alpha * task_vector_1[name]
                        + beta * task_vector_2[name]
                    )

    def format_eta(self, avg_time: float, remaining: int) -> str:
        """Format ETA string.

        Args:
            avg_time: Average time per iteration
            remaining: Remaining iterations

        Returns:
            Formatted ETA string
        """
        eta_seconds = avg_time * remaining
        if eta_seconds > 60:
            return f"ETA: {eta_seconds / 60:.1f}m"
        else:
            return f"ETA: {eta_seconds:.0f}s"

    def start_timing(self) -> float:
        """Start timing the experiment.

        Returns:
            Start timestamp
        """
        current_time = time.time()
        self.start_time = current_time
        return current_time

    def end_timing(self) -> float:
        """End timing the experiment.

        Returns:
            End timestamp
        """
        current_time = time.time()
        self.end_time = current_time
        return current_time

    def get_duration(self) -> float:
        """Get experiment duration.

        Returns:
            Duration in seconds
        """
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def get_timing_metadata(self) -> Dict[str, Any]:
        """Get timing metadata for the experiment.

        Returns:
            Dictionary with timing information
        """
        duration = self.get_duration()
        return {
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else "",
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else "",
            "duration_seconds": duration,
        }

    def prepare_model(self) -> None:
        """Prepare model for evaluation (move to device, set eval mode)."""
        self.base_model = self.base_model.to(self.device)  # type: ignore[arg-type,assignment]
        self.base_model.eval()
