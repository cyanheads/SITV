"""
Task vector computation and manipulation service.

This module provides the TaskVectorService for computing and manipulating
task vectors from fine-tuned and base models.
"""

import torch
from transformers import PreTrainedModel
from typing import Dict

from sitv.utils.progress import ProgressTracker


class TaskVectorService:
    """Service for task vector operations.

    Task vectors represent the difference between a fine-tuned model and
    its base model: T = M_finetuned - M_base

    This service handles:
    - Computing task vectors from model pairs
    - Applying task vectors to base models
    - Computing task vector magnitudes
    - Handling device management for parameters
    """

    @staticmethod
    def compute(
        base_model: PreTrainedModel, finetuned_model: PreTrainedModel
    ) -> Dict[str, torch.Tensor]:
        """Compute task vector: T = M_finetuned - M_base

        Handles models on different devices by moving parameters to CPU
        before subtraction.

        Args:
            base_model: Base (pretrained) model
            finetuned_model: Fine-tuned model

        Returns:
            Dictionary mapping parameter names to task vector tensors (on CPU)

        Raises:
            RuntimeError: If either model has parameters on meta device

        Examples:
            >>> service = TaskVectorService()
            >>> task_vector = service.compute(base_model, finetuned_model)
            >>> magnitude = service.compute_magnitude(task_vector)
        """
        task_vector = {}

        # Get parameter list for progress tracking
        params_list = list(
            zip(base_model.named_parameters(), finetuned_model.named_parameters(), strict=True)
        )
        tracker = ProgressTracker(total=len(params_list))

        for (name, base_param), (_, ft_param) in params_list:
            tracker.start_iteration()

            # Check for meta device - this indicates improper materialization
            if ft_param.device.type == "meta":
                raise RuntimeError(
                    f"Parameter '{name}' in fine-tuned model is on meta device. "
                    "This happens with device_map='auto' and model offloading. "
                    "Load the fine-tuning model with device_map=None instead."
                )
            if base_param.device.type == "meta":
                raise RuntimeError(
                    f"Parameter '{name}' in base model is on meta device. "
                    "Load the model with device_map=None instead."
                )

            # Move both to CPU first to handle device mismatches (e.g., cuda vs cpu)
            base_cpu = base_param.detach().cpu()
            ft_cpu = ft_param.detach().cpu()
            task_vector[name] = (ft_cpu - base_cpu).clone()

            # Update progress (print every 10 params or for large models)
            tracker.end_iteration()
            if tracker.current % 10 == 0 or len(params_list) > 100:
                msg = f"  Parameter progress: {tracker.get_status()}"
                print(f"{msg:<80}", end="\r", flush=True)

        print()  # New line after completion
        return task_vector

    @staticmethod
    def compute_magnitude(task_vector: Dict[str, torch.Tensor]) -> float:
        """Compute L2 magnitude of task vector.

        Args:
            task_vector: Dictionary of parameter name to tensor mappings

        Returns:
            L2 norm (magnitude) of the flattened task vector

        Examples:
            >>> magnitude = TaskVectorService.compute_magnitude(task_vector)
            >>> print(f"Task vector magnitude: {magnitude:.4f}")
        """
        # Use torch operations for numerical stability and precision
        # Compute sum of squared norms in tensor space before converting to float
        total_norm_sq = torch.tensor(0.0, dtype=torch.float64)
        for param in task_vector.values():
            # Use float64 for accumulation to prevent precision loss
            param_norm_sq = torch.sum(param.double() ** 2)
            total_norm_sq += param_norm_sq

        return float(torch.sqrt(total_norm_sq).item())

    @staticmethod
    def apply(
        base_model: PreTrainedModel,
        task_vector: Dict[str, torch.Tensor],
        alpha: float,
        device: torch.device,
    ) -> PreTrainedModel:
        """Apply scaled task vector to base model: M(α) = M_base + α·T

        Args:
            base_model: Base model to modify
            task_vector: Task vector to apply
            alpha: Scaling factor
            device: Device to move model to

        Returns:
            Modified model with task vector applied

        Raises:
            ValueError: If alpha is not finite (NaN or Inf)

        Note:
            This modifies the model in-place but returns it for convenience.

        Examples:
            >>> modified_model = TaskVectorService.apply(
            ...     base_model, task_vector, alpha=1.5, device=device
            ... )
        """
        # Validate alpha is finite
        if not torch.isfinite(torch.tensor(alpha)):
            raise ValueError(
                f"Alpha must be finite, got {alpha}. NaN or Inf values are not allowed."
            )

        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in task_vector:
                    # Move task vector to same device as parameter
                    tv = task_vector[name].to(param.device)
                    param.add_(tv, alpha=alpha)

        return base_model

    @staticmethod
    def create_model_at_alpha(
        base_model: PreTrainedModel,
        task_vector: Dict[str, torch.Tensor],
        alpha: float,
        device: torch.device,
    ) -> PreTrainedModel:
        """Create a new model at M(α) = M_base + α·T without modifying base.

        Args:
            base_model: Base model (will not be modified)
            task_vector: Task vector to apply
            alpha: Scaling factor
            device: Device for the new model

        Returns:
            New model with task vector applied at scale alpha

        Note:
            This creates a deep copy of the base model first.

        Examples:
            >>> model_alpha = TaskVectorService.create_model_at_alpha(
            ...     base_model, task_vector, alpha=2.0, device=device
            ... )
        """
        import copy

        model = copy.deepcopy(base_model)
        return TaskVectorService.apply(model, task_vector, alpha, device)
