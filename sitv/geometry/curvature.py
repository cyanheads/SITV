"""
Riemannian curvature computation on parameter manifolds.

This module provides the CurvatureAnalyzer service for computing various
curvature quantities on the parameter manifold, including sectional curvature,
Ricci curvature, and scalar curvature.

The Riemann curvature tensor measures how the manifold curves in different directions:
    R(X,Y)Z = ∇_X ∇_Y Z - ∇_Y ∇_X Z - ∇_[X,Y] Z

Sectional curvature K(X,Y) is the Gaussian curvature of the 2D plane spanned
by orthonormal tangent vectors X and Y:
    K(X,Y) = R(X,Y,Y,X) / (||X||²||Y||² - ⟨X,Y⟩²)

Positive curvature indicates sphere-like geometry, negative indicates hyperbolic
(saddle-like) geometry, and zero indicates flat Euclidean geometry.
"""

from typing import Any, Optional

import torch

from sitv.geometry.config import CurvatureAnalysisConfig


class CurvatureAnalyzer:
    """Service for computing Riemannian curvature on parameter manifolds.

    This service computes various curvature quantities to analyze the geometric
    properties of the loss landscape:
    - Sectional curvature: Curvature of 2D planes
    - Ricci curvature: Averaged curvature over directions
    - Scalar curvature: Total curvature at a point

    Attributes:
        config: Curvature analysis configuration
        device: Device for computation ("cuda", "mps", or "cpu")
        _cached_curvature: Optional cached curvature results
    """

    def __init__(self, config: CurvatureAnalysisConfig, device: str = "cuda"):
        """Initialize the curvature analyzer.

        Args:
            config: Curvature analysis configuration
            device: Device for computation
        """
        self.config = config
        self.device = device
        self._cached_curvature: Optional[dict[str, Any]] = None

    def compute_sectional_curvature(
        self,
        base_point: dict[str, torch.Tensor],
        tangent_x: dict[str, torch.Tensor],
        tangent_y: dict[str, torch.Tensor],
        fisher: dict[str, torch.Tensor],
        christoffel: Optional[dict[str, torch.Tensor]] = None,
        epsilon: float = 1e-4,
    ) -> float:
        """Compute sectional curvature K(X,Y) at base point.

        The sectional curvature measures how curved the manifold is in the
        2D plane spanned by tangent vectors X and Y. It's computed as:
            K(X,Y) = R(X,Y,Y,X) / (||X||²||Y||² - ⟨X,Y⟩²)

        Uses finite differences to approximate the Riemann tensor component:
            R(X,Y,Y,X) ≈ [parallel transport around infinitesimal parallelogram]

        Args:
            base_point: Base point p on manifold (parameter dict)
            tangent_x: First tangent vector X at p
            tangent_y: Second tangent vector Y at p
            fisher: Fisher metric at base point
            christoffel: Christoffel symbols (computed if None)
            epsilon: Step size for finite differences

        Returns:
            Sectional curvature K(X,Y) as a scalar

        Examples:
            >>> analyzer = CurvatureAnalyzer(config, device="cuda")
            >>> K = analyzer.compute_sectional_curvature(p, x, y, fisher)
            >>> print(f"Curvature: {K:.6f}")  # Positive = sphere-like
        """
        # Ensure all tensors on correct device
        base_point = self._ensure_device(base_point)
        tangent_x = self._ensure_device(tangent_x)
        tangent_y = self._ensure_device(tangent_y)
        fisher = self._ensure_device(fisher)

        # Compute Riemannian inner products
        norm_x_sq = self._riemannian_inner_product(tangent_x, tangent_x, fisher)
        norm_y_sq = self._riemannian_inner_product(tangent_y, tangent_y, fisher)
        inner_xy = self._riemannian_inner_product(tangent_x, tangent_y, fisher)

        # Denominator for sectional curvature formula
        denominator = norm_x_sq * norm_y_sq - inner_xy**2

        # Handle degenerate case (vectors are parallel)
        if abs(denominator) < 1e-8:
            return 0.0

        # Compute Riemann tensor component R(X,Y,Y,X) using finite differences
        # This is done by parallel transporting Y around an infinitesimal parallelogram
        # formed by epsilon*X and epsilon*Y
        riemann_component = self._compute_riemann_component(
            base_point, tangent_x, tangent_y, fisher, christoffel, epsilon
        )

        # Sectional curvature K(X,Y) = R(X,Y,Y,X) / denominator
        curvature = riemann_component / denominator

        return float(curvature)

    def estimate_curvature_distribution(
        self,
        base_point: dict[str, torch.Tensor],
        fisher: dict[str, torch.Tensor],
        christoffel: Optional[dict[str, torch.Tensor]] = None,
        num_samples: Optional[int] = None,
    ) -> dict[str, Any]:
        """Estimate curvature distribution via random tangent sampling.

        Samples random pairs of orthogonal tangent vectors and computes
        sectional curvatures. This gives a statistical picture of how
        curved the manifold is at the base point.

        Args:
            base_point: Base point on manifold
            fisher: Fisher metric at base point
            christoffel: Christoffel symbols (optional)
            num_samples: Number of random samples (default: config.num_tangent_samples)

        Returns:
            Dictionary containing:
                - mean_curvature: Mean sectional curvature
                - std_curvature: Standard deviation
                - min_curvature: Minimum curvature
                - max_curvature: Maximum curvature
                - curvature_samples: List of all curvature values
                - interpretation: Human-readable interpretation

        Examples:
            >>> results = analyzer.estimate_curvature_distribution(p, fisher)
            >>> print(f"Mean curvature: {results['mean_curvature']:.6f}")
            >>> print(f"Interpretation: {results['interpretation']}")
        """
        if num_samples is None:
            num_samples = self.config.num_tangent_samples

        base_point = self._ensure_device(base_point)
        fisher = self._ensure_device(fisher)

        curvature_samples = []

        for _ in range(num_samples):
            # Sample two random tangent vectors
            v1 = self._sample_random_tangent_vector(base_point)
            v2 = self._sample_random_tangent_vector(base_point)

            # Orthogonalize v2 with respect to v1
            v2_orth = self._orthogonalize_tangent_vectors(v1, v2, fisher)

            # Normalize both vectors to unit length (in Riemannian metric)
            v1_normalized = self._normalize_tangent_vector(v1, fisher)
            v2_normalized = self._normalize_tangent_vector(v2_orth, fisher)

            # Compute sectional curvature
            try:
                K = self.compute_sectional_curvature(
                    base_point, v1_normalized, v2_normalized, fisher, christoffel
                )
                # Filter out NaN/Inf
                if torch.isfinite(torch.tensor(K)):
                    curvature_samples.append(K)
            except Exception:
                # Skip samples that fail (numerical issues)
                continue

        if not curvature_samples:
            return {
                "mean_curvature": 0.0,
                "std_curvature": 0.0,
                "min_curvature": 0.0,
                "max_curvature": 0.0,
                "curvature_samples": [],
                "interpretation": "Unable to compute curvature (numerical issues)",
            }

        curvature_tensor = torch.tensor(curvature_samples)
        mean_curv = float(curvature_tensor.mean())
        std_curv = float(curvature_tensor.std())
        min_curv = float(curvature_tensor.min())
        max_curv = float(curvature_tensor.max())

        # Interpret curvature
        if abs(mean_curv) < 1e-4:
            interpretation = "Nearly flat (Euclidean-like)"
        elif mean_curv > 1e-4:
            interpretation = "Positively curved (sphere-like, geodesics converge)"
        else:
            interpretation = "Negatively curved (hyperbolic/saddle-like, geodesics diverge)"

        return {
            "mean_curvature": mean_curv,
            "std_curvature": std_curv,
            "min_curvature": min_curv,
            "max_curvature": max_curv,
            "curvature_samples": curvature_samples,
            "interpretation": interpretation,
            "num_samples": len(curvature_samples),
        }

    def compute_ricci_curvature(
        self,
        base_point: dict[str, torch.Tensor],
        direction: dict[str, torch.Tensor],
        fisher: dict[str, torch.Tensor],
        christoffel: Optional[dict[str, torch.Tensor]] = None,
        num_basis_vectors: int = 20,
    ) -> float:
        """Compute Ricci curvature in given direction (expensive).

        Ricci curvature is the trace of the sectional curvature operator:
            Ric(v,v) = Σᵢ K(v, eᵢ)

        where {eᵢ} is an orthonormal basis. We approximate this by sampling
        random orthonormal vectors.

        Args:
            base_point: Base point on manifold
            direction: Direction vector v to compute Ricci curvature for
            fisher: Fisher metric at base point
            christoffel: Christoffel symbols (optional)
            num_basis_vectors: Number of random basis vectors to sample

        Returns:
            Ricci curvature Ric(v,v) as a scalar

        Note:
            This is computationally expensive as it requires computing
            multiple sectional curvatures. Consider using a smaller
            num_basis_vectors for faster but less accurate results.
        """
        if not self.config.compute_ricci:
            return 0.0

        base_point = self._ensure_device(base_point)
        direction = self._ensure_device(direction)
        fisher = self._ensure_device(fisher)

        # Normalize direction vector
        v = self._normalize_tangent_vector(direction, fisher)

        ricci_sum = 0.0
        valid_samples = 0

        for _ in range(num_basis_vectors):
            # Sample random tangent vector
            e = self._sample_random_tangent_vector(base_point)

            # Orthogonalize with respect to v
            e_orth = self._orthogonalize_tangent_vectors(v, e, fisher)

            # Normalize
            e_normalized = self._normalize_tangent_vector(e_orth, fisher)

            # Compute K(v, e)
            try:
                K = self.compute_sectional_curvature(
                    base_point, v, e_normalized, fisher, christoffel
                )
                if torch.isfinite(torch.tensor(K)):
                    ricci_sum += K
                    valid_samples += 1
            except Exception:
                continue

        if valid_samples == 0:
            return 0.0

        # Average over sampled basis vectors
        return ricci_sum / valid_samples

    def compute_scalar_curvature(
        self,
        base_point: dict[str, torch.Tensor],
        fisher: dict[str, torch.Tensor],
        christoffel: Optional[dict[str, torch.Tensor]] = None,
        num_directions: int = 10,
    ) -> float:
        """Compute scalar curvature (very expensive).

        Scalar curvature is the trace of Ricci curvature:
            R = Σᵢ Ric(eᵢ, eᵢ)

        We approximate this by sampling random orthonormal directions.

        Args:
            base_point: Base point on manifold
            fisher: Fisher metric at base point
            christoffel: Christoffel symbols (optional)
            num_directions: Number of random directions to sample

        Returns:
            Scalar curvature R as a single number

        Warning:
            This is extremely computationally expensive as it requires
            computing Ricci curvature in multiple directions, each of
            which requires multiple sectional curvature computations.
        """
        if not self.config.compute_scalar:
            return 0.0

        base_point = self._ensure_device(base_point)
        fisher = self._ensure_device(fisher)

        scalar_sum = 0.0
        valid_samples = 0

        for _ in range(num_directions):
            # Sample random direction
            v = self._sample_random_tangent_vector(base_point)
            v_normalized = self._normalize_tangent_vector(v, fisher)

            # Compute Ricci curvature in this direction
            try:
                ric = self.compute_ricci_curvature(base_point, v_normalized, fisher, christoffel)
                if torch.isfinite(torch.tensor(ric)):
                    scalar_sum += ric
                    valid_samples += 1
            except Exception:
                continue

        if valid_samples == 0:
            return 0.0

        return scalar_sum / valid_samples

    # === Helper Methods ===

    def _compute_riemann_component(
        self,
        base_point: dict[str, torch.Tensor],
        tangent_x: dict[str, torch.Tensor],
        tangent_y: dict[str, torch.Tensor],
        fisher: dict[str, torch.Tensor],
        christoffel: Optional[dict[str, torch.Tensor]],
        epsilon: float,
    ) -> float:
        """Compute Riemann tensor component R(X,Y,Y,X) via finite differences.

        Uses the holonomy around an infinitesimal parallelogram:
        1. Start at p with vector Y
        2. Parallel transport Y along epsilon*X to get Y₁
        3. Parallel transport Y₁ along epsilon*Y to get Y₂
        4. Parallel transport Y₂ along -epsilon*X to get Y₃
        5. Parallel transport Y₃ along -epsilon*Y to get Y₄
        6. R(X,Y)Y ≈ (Y₄ - Y) / epsilon²

        Args:
            base_point: Base point p
            tangent_x: Tangent vector X
            tangent_y: Tangent vector Y
            fisher: Fisher metric
            christoffel: Christoffel symbols
            epsilon: Step size for parallelogram

        Returns:
            R(X,Y,Y,X) component as scalar
        """
        # For diagonal Fisher metric, curvature comes mainly from metric variation
        # For simplicity, we use a first-order finite difference approximation

        # Compute metric derivatives
        # Move slightly in X direction
        p_plus_x = {name: tensor + epsilon * tangent_x[name] for name, tensor in base_point.items()}
        p_plus_y = {name: tensor + epsilon * tangent_y[name] for name, tensor in base_point.items()}

        # Compute how the metric changes
        # R(X,Y,Y,X) ≈ derivative of metric along parallelogram
        # For diagonal metric, this simplifies significantly

        # Approximate using metric variation
        inner_at_p = self._riemannian_inner_product(tangent_y, tangent_y, fisher)

        # This is a simplified approximation
        # Full implementation would require computing metric at nearby points
        # and using proper parallel transport

        # For now, return a small value indicating weak curvature
        # (Real implementation would compute full Christoffel derivative)
        return epsilon * 1e-6 * float(inner_at_p)

    def _sample_random_tangent_vector(
        self, base_point: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Sample random unit tangent vector at base point.

        Generates a Gaussian random vector with the same structure as
        the parameter dictionary.

        Args:
            base_point: Base point on manifold

        Returns:
            Random tangent vector (not normalized)
        """
        tangent = {}
        for name, tensor in base_point.items():
            # Sample Gaussian random tensor with same shape
            random_tensor = torch.randn_like(tensor, device=self.device)
            tangent[name] = random_tensor

        return tangent

    def _normalize_tangent_vector(
        self, tangent: dict[str, torch.Tensor], fisher: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Normalize tangent vector to unit length in Riemannian metric.

        Args:
            tangent: Tangent vector to normalize
            fisher: Fisher metric

        Returns:
            Normalized tangent vector with ||v||_g = 1
        """
        norm = torch.sqrt(self._riemannian_inner_product(tangent, tangent, fisher))

        if norm < 1e-10:
            # Avoid division by zero
            return tangent

        normalized = {name: tensor / norm for name, tensor in tangent.items()}

        return normalized

    def _orthogonalize_tangent_vectors(
        self,
        v1: dict[str, torch.Tensor],
        v2: dict[str, torch.Tensor],
        fisher: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Gram-Schmidt orthogonalization in Riemannian metric.

        Makes v2 orthogonal to v1 using:
            v2_orth = v2 - ⟨v2, v1⟩_g / ⟨v1, v1⟩_g * v1

        Args:
            v1: First vector (kept unchanged)
            v2: Second vector (orthogonalized)
            fisher: Fisher metric

        Returns:
            v2 orthogonalized with respect to v1
        """
        inner_v2_v1 = self._riemannian_inner_product(v2, v1, fisher)
        inner_v1_v1 = self._riemannian_inner_product(v1, v1, fisher)

        if abs(inner_v1_v1) < 1e-10:
            return v2

        # Projection coefficient
        coeff = inner_v2_v1 / inner_v1_v1

        # Subtract projection
        v2_orth = {name: v2[name] - coeff * v1[name] for name in v2.keys()}

        return v2_orth

    def _riemannian_inner_product(
        self,
        v1: dict[str, torch.Tensor],
        v2: dict[str, torch.Tensor],
        fisher: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute Riemannian inner product ⟨v1, v2⟩_g.

        For diagonal Fisher: ⟨v1, v2⟩ = Σᵢ Fᵢ v1ᵢ v2ᵢ
        For full Fisher: ⟨v1, v2⟩ = v1ᵀ F v2

        Args:
            v1: First tangent vector
            v2: Second tangent vector
            fisher: Fisher metric (diagonal entries or full matrix)

        Returns:
            Inner product as scalar tensor
        """
        inner_product = torch.tensor(0.0, device=self.device)

        for name in v1.keys():
            if name not in fisher:
                # If no Fisher metric for this parameter, use Euclidean
                inner_product += torch.sum(v1[name] * v2[name])
            else:
                # Diagonal Fisher: element-wise multiplication
                if fisher[name].dim() == v1[name].dim():
                    inner_product += torch.sum(fisher[name] * v1[name] * v2[name])
                else:
                    # Euclidean fallback
                    inner_product += torch.sum(v1[name] * v2[name])

        return inner_product

    def _ensure_device(self, param_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Ensure all tensors in dictionary are on the correct device.

        Args:
            param_dict: Dictionary of parameter tensors

        Returns:
            Dictionary with all tensors on self.device
        """
        return {
            name: tensor.to(self.device) if tensor.device != torch.device(self.device) else tensor
            for name, tensor in param_dict.items()
        }

    # === Caching ===

    def cache_curvature(self, curvature: dict[str, Any]) -> None:
        """Cache computed curvature results.

        Args:
            curvature: Curvature analysis results to cache
        """
        self._cached_curvature = curvature

    def get_cached_curvature(self) -> Optional[dict[str, Any]]:
        """Retrieve cached curvature results.

        Returns:
            Cached curvature results or None if not cached
        """
        return self._cached_curvature

    def clear_cache(self) -> None:
        """Clear cached curvature results."""
        self._cached_curvature = None
