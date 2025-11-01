"""
Riemannian task vector computation using geodesics and Fisher metric.

This module extends the standard TaskVectorService with Riemannian geometry,
using the Fisher Information Matrix as the metric tensor and geodesic paths
for interpolation.
"""

from typing import Dict, Optional

import torch
from transformers import PreTrainedModel

from sitv.core.task_vector import TaskVectorService
from sitv.geometry.config import GeometryConfig
from sitv.geometry.geodesic import GeodesicIntegrator
from sitv.geometry.metric import FisherMetricService


class GeodesicTaskVectorService:
    """Service for Riemannian task vector operations.

    This service extends task vector operations with proper Riemannian geometry:
    - Uses Fisher metric for distances and norms
    - Computes task vectors via parallel transport (when enabled)
    - Applies task vectors via geodesic exponential map

    Attributes:
        config: Geometry configuration
        tokenizer: HuggingFace tokenizer for Fisher computation
        device: Device for computation
        fisher_service: Fisher metric service
        geodesic_integrator: Geodesic integration service
    """

    def __init__(
        self,
        config: GeometryConfig,
        tokenizer,
        device: str = "cuda"
    ):
        """Initialize the geodesic task vector service.

        Args:
            config: Geometry configuration
            tokenizer: HuggingFace tokenizer
            device: Device for computation
        """
        self.config = config
        self.tokenizer = tokenizer
        self.device = device

        # Initialize Fisher metric service
        self.fisher_service = FisherMetricService(
            tokenizer=tokenizer,
            device=device,
            approximation_type=config.metric_type,
            num_samples=config.fisher_approximation.num_samples,
            eigenvalue_floor=config.fisher_approximation.eigenvalue_floor
        )

        # Initialize geodesic integrator
        self.geodesic_integrator = GeodesicIntegrator(
            config=config.geodesic_integration,
            device=device
        )

        # Cache for Fisher metrics
        self._fisher_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    def compute(
        self,
        base_model: PreTrainedModel,
        finetuned_model: PreTrainedModel,
        data_texts: Optional[list[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute task vector using Riemannian geometry.

        In Euclidean space: T = M_ft - M_base
        In Riemannian space: T = log_M_base(M_ft) via parallel transport

        Args:
            base_model: Base (pretrained) model
            finetuned_model: Fine-tuned model
            data_texts: Optional data for parallel transport (experimental)

        Returns:
            Dictionary mapping parameter names to task vector tensors

        Examples:
            >>> service = GeodesicTaskVectorService(config, tokenizer, device)
            >>> task_vector = service.compute(base_model, finetuned_model)
        """
        # Get base parameters as dict
        base_params = {
            name: param.detach().cpu().clone()
            for name, param in base_model.named_parameters()
        }

        # Get finetuned parameters as dict
        ft_params = {
            name: param.detach().cpu().clone()
            for name, param in finetuned_model.named_parameters()
        }

        # If parallel transport enabled and data provided, use Riemannian log map
        if self.config.parallel_transport and data_texts is not None:
            # Compute Fisher metric at base point
            fisher = self.get_or_compute_fisher(base_model, data_texts, "base")

            # Use logarithm map to get tangent vector
            task_vector = self.geodesic_integrator.log_map(
                base_params, ft_params, fisher
            )
        else:
            # Fall back to Euclidean subtraction (which is the log map in flat space)
            task_vector = TaskVectorService.compute(base_model, finetuned_model)

        return task_vector

    def compute_magnitude(
        self,
        task_vector: Dict[str, torch.Tensor],
        fisher: Optional[Dict[str, torch.Tensor]] = None
    ) -> float:
        """Compute Riemannian magnitude of task vector.

        In Euclidean space: ||T|| = √(Σ||T_i||²)
        In Riemannian space: ||T||_g = √(Σ F_ij T^i T^j)

        Args:
            task_vector: Task vector dictionary
            fisher: Optional Fisher metric (uses Euclidean if None)

        Returns:
            Riemannian norm of task vector

        Examples:
            >>> magnitude = service.compute_magnitude(task_vector, fisher)
        """
        if fisher is None or not self.config.use_riemannian:
            # Fall back to Euclidean norm
            return TaskVectorService.compute_magnitude(task_vector)

        # Use Riemannian norm with Fisher metric
        return self.fisher_service.compute_riemannian_norm(task_vector, fisher)

    def apply_geodesic(
        self,
        base_model: PreTrainedModel,
        task_vector: Dict[str, torch.Tensor],
        alpha: float,
        fisher: Optional[Dict[str, torch.Tensor]] = None,
        christoffel: Optional[Dict[str, torch.Tensor]] = None
    ) -> PreTrainedModel:
        """Apply task vector via geodesic exponential map.

        In Euclidean space: M(α) = M_base + α·T
        In Riemannian space: M(α) = exp_M_base(α·T)

        Args:
            base_model: Base model
            task_vector: Task vector to apply
            alpha: Scaling factor
            fisher: Optional Fisher metric
            christoffel: Optional Christoffel symbols

        Returns:
            Modified model at geodesic point M(α)

        Note:
            This modifies the model in-place.

        Examples:
            >>> model_alpha = service.apply_geodesic(
            ...     base_model, task_vector, alpha=1.5, fisher=fisher
            ... )
        """
        # Get base parameters as dict
        base_params = {
            name: param.data.clone()
            for name, param in base_model.named_parameters()
        }

        # Compute geodesic endpoint
        if self.config.use_geodesics and fisher is not None:
            # Use Riemannian exponential map
            new_params = self.geodesic_integrator.exponential_map(
                base_point=base_params,
                tangent_vector=task_vector,
                t=alpha,
                fisher_metric=fisher,
                christoffel=christoffel
            )
        else:
            # Fall back to Euclidean straight line
            new_params = self.geodesic_integrator.exponential_map(
                base_point=base_params,
                tangent_vector=task_vector,
                t=alpha,
                fisher_metric=None,
                christoffel=None
            )

        # Apply new parameters to model
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in new_params:
                    param.data.copy_(new_params[name].to(param.device))

        return base_model

    def get_or_compute_fisher(
        self,
        model: PreTrainedModel,
        data_texts: list[str],
        cache_key: str
    ) -> Dict[str, torch.Tensor]:
        """Get Fisher metric from cache or compute it.

        Args:
            model: Model to compute Fisher for
            data_texts: Data samples for Fisher estimation
            cache_key: Key for caching (e.g., "base", "finetuned")

        Returns:
            Fisher Information Matrix

        Examples:
            >>> fisher = service.get_or_compute_fisher(model, texts, "base")
        """
        # Check cache if enabled
        if self.config.cache_metric and cache_key in self._fisher_cache:
            return self._fisher_cache[cache_key]

        # Compute Fisher matrix
        fisher = self.fisher_service.compute_fisher_information_matrix(
            model=model,
            texts=data_texts,
            batch_size=8
        )

        # Cache if enabled
        if self.config.cache_metric:
            self._fisher_cache[cache_key] = fisher

        return fisher

    def compute_christoffel(
        self,
        model: PreTrainedModel,
        fisher: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute Christoffel symbols from Fisher metric.

        Args:
            model: Model (for parameter structure)
            fisher: Fisher metric

        Returns:
            Christoffel symbols

        Examples:
            >>> christoffel = service.compute_christoffel(model, fisher)
        """
        return self.fisher_service.compute_christoffel_symbols(model, fisher)

    def clear_cache(self) -> None:
        """Clear cached Fisher metrics."""
        self._fisher_cache.clear()
        self.fisher_service.clear_cache()
