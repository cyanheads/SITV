"""
Fisher Information Matrix computation and Riemannian metric operations.

This module provides the FisherMetricService for computing the Fisher Information
Matrix (FIM) as a Riemannian metric on the parameter manifold. The Fisher metric
defines the local geometry of the statistical model.

The Fisher Information Matrix is defined as:
    F_ij = E[∂log p(x|θ)/∂θ_i × ∂log p(x|θ)/∂θ_j]

where p(x|θ) is the model's probability distribution and E[·] is expectation
over the data distribution.
"""

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from sitv.geometry.config import FisherApproximationType
from sitv.utils.progress import ProgressTracker

if TYPE_CHECKING:
    from sitv.geometry.config import ChristoffelComputationConfig


class FisherMetricService:
    """Service for computing Fisher Information Matrix and Riemannian metrics.

    This service supports three approximation levels:
    - Diagonal: O(n) memory, fast, assumes parameter independence
    - KFAC: O(k²) for k blocks, practical block-diagonal approximation
    - Full: O(n²) memory, exact but expensive for large models

    Attributes:
        tokenizer: HuggingFace tokenizer for text preprocessing
        device: Device to run computations on
        approximation_type: Type of Fisher approximation
        num_samples: Number of data samples for FIM computation
        eigenvalue_floor: Minimum eigenvalue to ensure positive-definiteness
        max_length: Maximum sequence length for tokenization
    """

    def __init__(
        self,
        tokenizer,
        device: str = "cuda",
        approximation_type: FisherApproximationType = FisherApproximationType.DIAGONAL,
        num_samples: int = 1000,
        eigenvalue_floor: float = 1e-6,
        max_length: int = 512
    ):
        """Initialize the Fisher metric service.

        Args:
            tokenizer: HuggingFace tokenizer
            device: Device for computation ("cuda", "mps", or "cpu")
            approximation_type: Type of Fisher approximation to use
            num_samples: Number of samples for FIM estimation
            eigenvalue_floor: Minimum eigenvalue for numerical stability
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.device = device
        self.approximation_type = approximation_type
        self.num_samples = num_samples
        self.eigenvalue_floor = eigenvalue_floor
        self.max_length = max_length

        # Cache for computed Fisher matrices
        self._cached_fisher: dict[str, torch.Tensor] | None = None

    def compute_fisher_information_matrix(
        self,
        model: nn.Module,
        texts: list[str],
        batch_size: int = 8
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        """Compute Fisher Information Matrix for model parameters.

        The Fisher matrix is computed as the expected outer product of
        gradients of log-likelihood:
            F = E[∇log p(x|θ) ∇log p(x|θ)ᵀ]

        Args:
            model: Model to compute Fisher for
            texts: Dataset samples for Fisher estimation
            batch_size: Batch size for gradient computation

        Returns:
            Dictionary mapping parameter names to Fisher matrices:
            - Diagonal: 1D tensor (diagonal elements)
            - KFAC: Tuple of (A, B) Kronecker factors
            - Full: 2D tensor (full covariance matrix)

        Examples:
            >>> fisher_service = FisherMetricService(tokenizer, device="cuda")
            >>> fisher = fisher_service.compute_fisher_information_matrix(model, texts)
        """
        if self.approximation_type == FisherApproximationType.EUCLIDEAN:
            # Return identity (no actual Fisher computation needed)
            return cast(dict[str, torch.Tensor | dict[str, Any]], self._identity_metric(model))

        # Sample subset of data if needed
        if len(texts) > self.num_samples:
            import random
            texts = random.sample(texts, self.num_samples)

        model.eval()

        if self.approximation_type == FisherApproximationType.DIAGONAL:
            return cast(dict[str, torch.Tensor | dict[str, Any]], self._compute_diagonal_fisher(model, texts, batch_size))
        elif self.approximation_type == FisherApproximationType.KFAC:
            return cast(dict[str, torch.Tensor | dict[str, Any]], self._compute_kfac_fisher(model, texts, batch_size))
        elif self.approximation_type == FisherApproximationType.FULL:
            return self._compute_full_fisher(model, texts, batch_size)
        else:
            raise ValueError(f"Unknown approximation type: {self.approximation_type}")

    def _identity_metric(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Return identity metric (Euclidean space).

        Args:
            model: Model to get parameter shapes for

        Returns:
            Dictionary with ones for each parameter (diagonal identity)
        """
        fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.ones_like(param.data)
        return fisher

    def _compute_diagonal_fisher(
        self,
        model: nn.Module,
        texts: list[str],
        batch_size: int
    ) -> dict[str, torch.Tensor]:
        """Compute diagonal Fisher approximation.

        Diagonal Fisher assumes parameter independence:
            F_ii = E[(∂log p/∂θ_i)²]

        This is the variance of gradients for each parameter.

        Args:
            model: Model to compute Fisher for
            texts: Dataset samples
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping parameter names to diagonal Fisher (1D tensors)
        """
        # Initialize accumulators for squared gradients
        fisher_diag = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_diag[name] = torch.zeros_like(param.data)

        count = 0

        # Initialize progress tracker
        total_batches = (len(texts) + batch_size - 1) // batch_size
        tracker = ProgressTracker(total=total_batches)

        # Accumulate gradients over batches
        for i in range(0, len(texts), batch_size):
            tracker.start_iteration()
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Zero gradients
            model.zero_grad()

            # Forward pass with gradient computation
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Backward pass to get gradients
            loss.backward()

            # Accumulate squared gradients (diagonal Fisher)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_diag[name] += param.grad.data ** 2

            count += len(batch_texts)

            # Update progress
            tracker.end_iteration()
            msg = f"  Batch progress: {tracker.get_status()}"
            print(f"{msg:<80}", end='\r', flush=True)

        print()  # New line after completion

        # Average and add floor for numerical stability
        for name in fisher_diag:
            fisher_diag[name] = (fisher_diag[name] / count) + self.eigenvalue_floor

        return fisher_diag

    def _compute_kfac_fisher(
        self,
        model: nn.Module,
        texts: list[str],
        batch_size: int
    ) -> dict[str, torch.Tensor]:
        """Compute KFAC (Kronecker-Factored) Fisher approximation.

        For linear layers W: F ≈ A ⊗ B where:
        - A is the input activation covariance
        - B is the gradient covariance

        This gives block-diagonal structure with much lower memory cost.

        Args:
            model: Model to compute Fisher for
            texts: Dataset samples
            batch_size: Batch size for processing

        Returns:
            Dictionary with Fisher approximations per layer

        Note:
            This is a simplified KFAC implementation. Full KFAC requires
            tracking activation statistics during forward pass.
        """
        # For now, fall back to diagonal approximation
        # Full KFAC requires forward hooks to capture activations
        # TODO: Implement proper KFAC with activation statistics
        return self._compute_diagonal_fisher(model, texts, batch_size)

    def _compute_full_fisher(
        self,
        model: nn.Module,
        texts: list[str],
        batch_size: int
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        """Compute full Fisher Information Matrix.

        WARNING: This computes the full O(n²) covariance matrix.
        Only use for small models or research purposes.

        Args:
            model: Model to compute Fisher for
            texts: Dataset samples
            batch_size: Batch size for processing

        Returns:
            Dictionary with full Fisher matrices per parameter

        Raises:
            MemoryError: If model is too large for full Fisher
        """
        # Flatten all parameters into a single vector
        param_shapes: dict[str, tuple[int, ...]] = {}
        param_list = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_shapes[name] = param.shape
                param_list.append(param.view(-1))

        total_params = sum(p.numel() for p in param_list)

        # Warn if too many parameters
        if total_params > 1e7:  # 10M parameters
            raise MemoryError(
                f"Model has {total_params:,} parameters. "
                f"Full Fisher matrix would require {total_params**2 * 4 / 1e9:.1f} GB. "
                f"Use DIAGONAL or KFAC approximation instead."
            )

        # Initialize Fisher matrix accumulator
        fisher_full = torch.zeros(
            (total_params, total_params),
            device=self.device,
            dtype=torch.float32
        )

        count = 0

        # Initialize progress tracker
        total_batches = (len(texts) + batch_size - 1) // batch_size
        tracker = ProgressTracker(total=total_batches)

        # Accumulate gradient outer products
        for i in range(0, len(texts), batch_size):
            tracker.start_iteration()
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            model.zero_grad()
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()

            # Flatten gradients into single vector
            grad_vector = []
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_vector.append(param.grad.view(-1))

            grad_flat = torch.cat(grad_vector)

            # Outer product: g g^T
            fisher_full += torch.outer(grad_flat, grad_flat)

            count += len(batch_texts)

            # Update progress
            tracker.end_iteration()
            msg = f"  Batch progress: {tracker.get_status()}"
            print(f"{msg:<80}", end='\r', flush=True)

        print()  # New line after completion

        # Average and add regularization
        fisher_full = fisher_full / count
        fisher_full += torch.eye(total_params, device=self.device) * self.eigenvalue_floor

        # Split back into per-parameter matrices
        # For simplicity, return as single matrix with metadata
        fisher_dict: dict[str, torch.Tensor | dict[str, Any]] = {
            "_full_matrix": fisher_full,
            "_param_shapes": param_shapes
        }
        return fisher_dict

    def compute_riemannian_norm(
        self,
        vector: dict[str, torch.Tensor],
        fisher: dict[str, torch.Tensor]
    ) -> float:
        """Compute Riemannian norm ||v||_g = sqrt(v^T G v).

        Args:
            vector: Dictionary of parameter tensors (e.g., task vector)
            fisher: Fisher metric (from compute_fisher_information_matrix)

        Returns:
            Riemannian norm (scalar)

        Examples:
            >>> norm = fisher_service.compute_riemannian_norm(task_vector, fisher)
        """
        # Ensure vector and Fisher are on the same device
        # Get device from first Fisher tensor
        if fisher and self.approximation_type != FisherApproximationType.EUCLIDEAN:
            first_fisher_key = next(iter(fisher.keys()))
            if first_fisher_key != "_full_matrix":
                fisher_device = fisher[first_fisher_key].device
            else:
                fisher_device = fisher["_full_matrix"].device

            # Move vector to Fisher's device if needed
            vector = {
                name: tensor.to(fisher_device) if tensor.device != fisher_device else tensor
                for name, tensor in vector.items()
            }

        if self.approximation_type == FisherApproximationType.EUCLIDEAN:
            # Euclidean norm: ||v||² = Σ ||v_i||²
            norm_sq = sum(
                torch.sum(vector[name] ** 2).item()
                for name in vector
            )
            return float(norm_sq ** 0.5)

        elif self.approximation_type in [
            FisherApproximationType.DIAGONAL,
            FisherApproximationType.KFAC
        ]:
            # Diagonal metric: ||v||² = Σ v_i² F_ii
            norm_sq = sum(
                torch.sum((vector[name] ** 2) * fisher[name]).item()
                for name in vector
                if name in fisher
            )
            return float(norm_sq ** 0.5)

        elif self.approximation_type == FisherApproximationType.FULL:
            # Full metric: ||v||² = v^T F v
            fisher_matrix = fisher["_full_matrix"]

            # Flatten vector
            v_flat = torch.cat([vector[name].view(-1) for name in vector])

            # Compute v^T F v
            norm_sq = torch.dot(v_flat, torch.mv(fisher_matrix, v_flat)).item()
            return float(max(0, norm_sq) ** 0.5)  # max for numerical safety

        else:
            raise ValueError(f"Unknown approximation type: {self.approximation_type}")

    def compute_riemannian_distance(
        self,
        params1: dict[str, torch.Tensor],
        params2: dict[str, torch.Tensor],
        fisher: dict[str, torch.Tensor]
    ) -> float:
        """Compute Riemannian distance between two parameter sets.

        This is an approximation using the metric at params1.
        True geodesic distance requires integration along the geodesic.

        Args:
            params1: First parameter dictionary
            params2: Second parameter dictionary
            fisher: Fisher metric at params1

        Returns:
            Approximate Riemannian distance

        Examples:
            >>> dist = fisher_service.compute_riemannian_distance(
            ...     base_params, finetuned_params, fisher
            ... )
        """
        # Compute difference vector
        diff = {}
        for name in params1:
            if name in params2:
                diff[name] = params2[name] - params1[name]

        # Return norm of difference
        return self.compute_riemannian_norm(diff, fisher)

    def compute_christoffel_symbols(
        self,
        model: nn.Module,
        fisher: dict[str, torch.Tensor]
    ) -> dict:
        """Compute Christoffel symbols from Fisher metric.

        Christoffel symbols Γᵏᵢⱼ define the Levi-Civita connection:
            Γᵏᵢⱼ = (1/2) gᵏˡ (∂g_jl/∂xⁱ + ∂g_il/∂xʲ - ∂g_ij/∂xˡ)

        For constant Fisher metric, Christoffel symbols are zero (flat space).
        For varying Fisher, we need to compute metric derivatives.

        Args:
            model: Model (for parameter structure)
            fisher: Fisher metric

        Returns:
            Dictionary with Christoffel symbols (simplified representation)

        Note:
            For diagonal Fisher that varies slowly, Christoffel symbols are
            approximately zero. This is a placeholder for more sophisticated
            implementations that track metric variation.
        """
        # Placeholder: For now, assume constant Fisher (Christoffel = 0)
        # Full implementation requires computing ∂F/∂θ numerically
        christoffel = {}
        for name in fisher:
            if not name.startswith("_"):  # Skip metadata keys
                christoffel[name] = torch.zeros_like(fisher[name])

        return christoffel

    def compute_christoffel_symbols_finite_diff(
        self,
        model: nn.Module,
        base_params: dict[str, torch.Tensor],
        data_texts: list[str],
        epsilon: float = 1e-3,
        batch_size: int = 8,
        config: "ChristoffelComputationConfig | None" = None
    ) -> dict[str, torch.Tensor]:
        """Compute Christoffel symbols via finite differences of Fisher metric.

        This method computes the metric derivatives ∂F/∂θ numerically and uses
        them to compute the Christoffel symbols:
            Γᵏᵢᵢ ≈ (1/2) F⁻¹ (∂F_ii/∂θ_k)

        For diagonal Fisher metric, we use the simplified formula for diagonal
        Christoffel symbols where only the diagonal components are non-zero.

        Args:
            model: Neural network model
            base_params: Current parameter values (dict of tensors)
            data_texts: Dataset samples for Fisher computation
            epsilon: Finite difference step size (default: 1e-3)
            batch_size: Batch size for Fisher computation
            config: Configuration for Christoffel computation (filtering, sampling, etc.)

        Returns:
            Dictionary mapping parameter names to Christoffel symbol tensors

        Note:
            This is computationally expensive as it requires recomputing the
            Fisher matrix for perturbed parameters. The config parameter allows
            filtering parameters (e.g., skip vision_tower, skip frozen params) to
            significantly reduce computation time.

        Mathematical Note:
            For a diagonal metric g_ii(θ), the non-zero Christoffel symbols are:
                Γᵏᵢᵢ = (1/2g_kk) ∂g_ii/∂θ_k

            We approximate the derivative using central differences:
                ∂F_ii/∂θ_k ≈ [F_ii(θ + ε·e_k) - F_ii(θ - ε·e_k)] / (2ε)

            where e_k is the k-th standard basis vector (directional perturbation).

        Examples:
            >>> fisher_service = FisherMetricService(tokenizer, device="cuda")
            >>> christoffel = fisher_service.compute_christoffel_symbols_finite_diff(
            ...     model, base_params, data_texts
            ... )
        """
        # Import here to avoid circular dependency
        from sitv.geometry.config import ChristoffelComputationConfig

        # Use default config if none provided
        if config is None:
            config = ChristoffelComputationConfig()

        # Get model parameters to check requires_grad status
        model_params = {n: p for n, p in model.named_parameters()}

        # Get Fisher at base point
        F_base = self.compute_fisher_information_matrix(model, data_texts, batch_size)

        christoffel = {}

        # For diagonal Fisher, we compute simplified Christoffel symbols
        # Γᵏᵢᵢ ≈ (1/2) F_kk⁻¹ (∂F_ii/∂θ_k)
        #
        # Simplification: For diagonal metric with slow variation, we approximate
        # using forward differences in random directions to detect metric curvature

        # Filter parameters based on config
        params_to_process = []
        stats = {
            "total": 0,
            "filtered_vision_tower": 0,
            "filtered_frozen": 0,
            "filtered_internal": 0,
            "processed": 0,
        }

        for name, param in base_params.items():
            stats["total"] += 1

            # Skip internal metadata keys
            if name not in F_base or name.startswith("_"):
                stats["filtered_internal"] += 1
                continue

            # Skip vision_tower parameters if configured
            if config.skip_vision_tower and "vision_tower" in name:
                stats["filtered_vision_tower"] += 1
                christoffel[name] = torch.zeros_like(param.data)
                continue

            # Skip frozen parameters if configured
            if config.skip_frozen and name in model_params:
                if not model_params[name].requires_grad:
                    stats["filtered_frozen"] += 1
                    christoffel[name] = torch.zeros_like(param.data)
                    continue

            params_to_process.append((name, param))

        stats["processed"] = len(params_to_process)

        # Apply max_parameters limit if specified
        if config.max_parameters is not None and len(params_to_process) > config.max_parameters:
            params_to_process = params_to_process[:config.max_parameters]
            stats["processed"] = config.max_parameters

        # Apply parameter sampling if fraction < 1.0
        if config.parameter_sample_fraction < 1.0:
            import random
            sample_size = max(1, int(len(params_to_process) * config.parameter_sample_fraction))
            params_to_process = random.sample(params_to_process, sample_size)
            stats["processed"] = sample_size

        # Log filtering statistics
        print("\n  Computing Christoffel symbols to detect curvature...")
        print(f"  Total parameters: {stats['total']}")
        print(f"  Filtered (vision_tower): {stats['filtered_vision_tower']}")
        print(f"  Filtered (frozen): {stats['filtered_frozen']}")
        print(f"  Filtered (internal): {stats['filtered_internal']}")
        print(f"  Processing: {stats['processed']} parameters\n")

        # Count parameters to process for progress tracking
        tracker = ProgressTracker(total=stats["processed"])

        for name, param in params_to_process:

            tracker.start_iteration()

            # Initialize Christoffel for this parameter
            christoffel[name] = torch.zeros_like(param.data)

            # Save original parameter
            original_param = param.data.clone()

            try:
                # Compute metric derivative via forward finite difference
                # Perturb in a random direction (we use sign of random noise)
                # to detect if Fisher varies
                with torch.no_grad():
                    # Create a small random perturbation
                    # Use sign to ensure we're moving in parameter space
                    perturbation = epsilon * torch.randn_like(param.data).sign_()
                    param.data.add_(perturbation)

                # Recompute Fisher at perturbed point
                F_perturbed = self.compute_fisher_information_matrix(
                    model, data_texts, batch_size
                )

                # Extract tensor values (skip metadata keys like "_full_matrix")
                F_base_val = F_base[name]
                F_perturbed_val = F_perturbed[name]
                assert isinstance(F_base_val, torch.Tensor), f"Expected tensor for {name}"
                assert isinstance(F_perturbed_val, torch.Tensor), f"Expected tensor for {name}"

                # Compute derivative of Fisher: ∂F/∂θ ≈ (F_new - F_old) / ε
                dF_dtheta = (F_perturbed_val - F_base_val) / epsilon

                # Simplified Christoffel for diagonal metric:
                # Γ ≈ (1/2) F⁻¹ (∂F/∂θ)
                # Add small epsilon to prevent division by zero
                christoffel[name] = 0.5 * dF_dtheta / (F_base_val + 1e-8)

            finally:
                # Restore original parameters
                with torch.no_grad():
                    param.data.copy_(original_param)

                # Update progress
                tracker.end_iteration()
                param_shape = "x".join(str(d) for d in param.shape)

                # Truncate long parameter names for better display
                display_name = name if len(name) <= 60 else f"...{name[-57:]}"

                # Format with padding to clear previous longer lines
                msg = (
                    f"  Overall: [{tracker.current}/{tracker.total}] {tracker.get_status()} "
                    f"| Current: {display_name} (shape: {param_shape})"
                )
                print(f"{msg:<150}", end='\r', flush=True)

        print()  # New line after completion
        return christoffel

    def cache_fisher(self, fisher: dict[str, torch.Tensor]) -> None:
        """Cache computed Fisher matrix for reuse.

        Args:
            fisher: Fisher matrix to cache
        """
        self._cached_fisher = fisher

    def get_cached_fisher(self) -> dict[str, torch.Tensor] | None:
        """Retrieve cached Fisher matrix.

        Returns:
            Cached Fisher matrix, or None if not cached
        """
        return self._cached_fisher

    def clear_cache(self) -> None:
        """Clear cached Fisher matrix."""
        self._cached_fisher = None
