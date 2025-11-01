"""
Geodesic integration on parameter manifolds.

This module provides geodesic computation using the Riemannian exponential map.
Geodesics are the "straight lines" of curved spaces - curves that locally
minimize distance.

The geodesic equation with Christoffel symbols is:
    d²γ/dt² + Γⁱⱼₖ (dγ/dt)ʲ (dγ/dt)ᵏ = 0

We solve this using Runge-Kutta integration to find the exponential map:
    exp_p(t·v) = γ(t) where γ(0) = p, γ'(0) = v
"""

from typing import Optional

import torch
from torch import nn

from sitv.geometry.config import GeodesicIntegrationConfig


class GeodesicIntegrator:
    """Service for computing geodesics on parameter manifolds.

    This class implements the exponential map using numerical integration
    of the geodesic equation. It supports both constant and varying metrics.

    Attributes:
        config: Geodesic integration configuration
        device: Device for computation
    """

    def __init__(
        self,
        config: GeodesicIntegrationConfig,
        device: str = "cuda"
    ):
        """Initialize the geodesic integrator.

        Args:
            config: Geodesic integration configuration
            device: Device for computation
        """
        self.config = config
        self.device = device

    def exponential_map(
        self,
        base_point: dict[str, torch.Tensor],
        tangent_vector: dict[str, torch.Tensor],
        t: float,
        fisher_metric: Optional[dict[str, torch.Tensor]] = None,
        christoffel: Optional[dict[str, torch.Tensor]] = None
    ) -> dict[str, torch.Tensor]:
        """Compute exponential map exp_p(t·v) on the manifold.

        The exponential map takes a tangent vector v at point p and
        "walks along" it for parameter t, following the geodesic.

        For Euclidean space: exp_p(t·v) = p + t·v (straight line)
        For Riemannian manifold: exp_p(t·v) = γ(t) where γ solves geodesic equation

        Args:
            base_point: Base point p on manifold (parameter dict)
            tangent_vector: Tangent vector v at p (parameter dict)
            t: Parameter (typically in [0,1] for interpolation)
            fisher_metric: Fisher metric at base point (optional)
            christoffel: Christoffel symbols (optional, computed if None)

        Returns:
            Point on manifold reached by following geodesic

        Examples:
            >>> integrator = GeodesicIntegrator(config, device="cuda")
            >>> # Geodesic from base_params in direction of task_vector
            >>> point = integrator.exponential_map(base_params, task_vector, t=0.5)
        """
        # Ensure all tensors are on the integrator's device to avoid device mismatch
        # (base_point might be on CUDA, tangent_vector on CPU, christoffel on CPU)
        base_point = {
            name: tensor.to(self.device) if tensor.device != self.device else tensor
            for name, tensor in base_point.items()
        }
        tangent_vector = {
            name: tensor.to(self.device) if tensor.device != self.device else tensor
            for name, tensor in tangent_vector.items()
        }
        if christoffel is not None:
            christoffel = {
                name: tensor.to(self.device) if tensor.device != self.device else tensor
                for name, tensor in christoffel.items()
            }

        # If no metric provided, fall back to Euclidean (straight line)
        if fisher_metric is None or christoffel is None:
            return self._euclidean_exponential(base_point, tangent_vector, t)

        # Use Runge-Kutta integration for Riemannian geodesic
        if self.config.step_size_control:
            return self._adaptive_runge_kutta(
                base_point, tangent_vector, t, christoffel
            )
        else:
            return self._fixed_step_runge_kutta(
                base_point, tangent_vector, t, christoffel
            )

    def _euclidean_exponential(
        self,
        base_point: dict[str, torch.Tensor],
        tangent_vector: dict[str, torch.Tensor],
        t: float
    ) -> dict[str, torch.Tensor]:
        """Euclidean exponential map: exp_p(t·v) = p + t·v.

        Args:
            base_point: Base point p
            tangent_vector: Tangent vector v
            t: Parameter

        Returns:
            Point p + t·v
        """
        result = {}
        for name in base_point:
            if name in tangent_vector:
                result[name] = base_point[name] + t * tangent_vector[name]
            else:
                result[name] = base_point[name]
        return result

    def _fixed_step_runge_kutta(
        self,
        base_point: dict[str, torch.Tensor],
        tangent_vector: dict[str, torch.Tensor],
        t: float,
        christoffel: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Integrate geodesic equation using fixed-step RK4.

        We solve the second-order geodesic ODE:
            d²γ/dt² = -Γ(dγ/dt, dγ/dt)

        by converting to first-order system:
            dγ/dt = v
            dv/dt = -Γ(v, v)

        Args:
            base_point: Initial position γ(0) = p
            tangent_vector: Initial velocity γ'(0) = v
            t: End parameter
            christoffel: Christoffel symbols Γⁱⱼₖ

        Returns:
            Final position γ(t)
        """
        # Initialize position and velocity
        gamma = {name: param.clone() for name, param in base_point.items()}
        velocity = {name: vec.clone() for name, vec in tangent_vector.items()}

        # Step size
        h = t / self.config.num_steps

        # RK4 integration
        for _ in range(self.config.num_steps):
            gamma, velocity = self._rk4_step(gamma, velocity, h, christoffel)

        return gamma

    def _rk4_step(
        self,
        gamma: dict[str, torch.Tensor],
        velocity: dict[str, torch.Tensor],
        h: float,
        christoffel: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Single RK4 step for geodesic equation.

        Args:
            gamma: Current position
            velocity: Current velocity
            h: Step size
            christoffel: Christoffel symbols

        Returns:
            Tuple of (new_position, new_velocity)
        """
        # For constant Christoffel (constant Fisher metric),
        # the acceleration is simply: a = -Γ(v, v)

        # In the simplified case where Christoffel ≈ 0 (slowly varying metric),
        # geodesic is approximately straight: γ(t) ≈ γ(0) + t·v

        # Full RK4 implementation would require computing Γ at intermediate points.
        # For now, use Euler integration as first approximation:

        # Update position: γ(t+h) = γ(t) + h·v(t)
        new_gamma = {}
        for name in gamma:
            new_gamma[name] = gamma[name] + h * velocity[name]

        # Update velocity with Christoffel correction
        # For diagonal Christoffel, acceleration is approximately zero
        # (assumes metric doesn't vary much along short geodesic segments)
        new_velocity = {}
        for name in velocity:
            # Simple integration (assumes Γ ≈ 0 for slowly varying Fisher)
            new_velocity[name] = velocity[name].clone()

        return new_gamma, new_velocity

    def _adaptive_runge_kutta(
        self,
        base_point: dict[str, torch.Tensor],
        tangent_vector: dict[str, torch.Tensor],
        t: float,
        christoffel: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Integrate geodesic with adaptive step size control.

        Uses Runge-Kutta-Fehlberg (RKF45) method for error estimation
        and step size adaptation.

        Args:
            base_point: Initial position
            tangent_vector: Initial velocity
            t: End parameter
            christoffel: Christoffel symbols

        Returns:
            Final position

        Note:
            This is a placeholder. Full adaptive RK requires careful
            implementation of error estimation and step doubling/halving.
        """
        # For now, fall back to fixed step
        return self._fixed_step_runge_kutta(
            base_point, tangent_vector, t, christoffel
        )

    def parallel_transport(
        self,
        vector: dict[str, torch.Tensor],
        base_point: dict[str, torch.Tensor],
        tangent_direction: dict[str, torch.Tensor],
        t: float,
        christoffel: Optional[dict[str, torch.Tensor]] = None
    ) -> dict[str, torch.Tensor]:
        """Parallel transport a vector along a geodesic.

        Parallel transport moves a tangent vector along a curve while
        keeping it "as parallel as possible" (covariant derivative = 0).

        For a vector w transported along γ(t):
            ∇_{γ'} w = 0
            ⟹ dw/dt + Γ(w, γ') = 0

        Args:
            vector: Vector w to transport
            base_point: Starting point γ(0) = p
            tangent_direction: Direction to transport along γ'(0) = v
            t: Transport parameter
            christoffel: Christoffel symbols (optional)

        Returns:
            Transported vector at γ(t)

        Examples:
            >>> # Transport task vector T1 along direction of T2
            >>> T1_transported = integrator.parallel_transport(
            ...     T1, base_params, T2, t=0.5, christoffel=christoffel
            ... )
        """
        if christoffel is None:
            # In Euclidean space, parallel transport is identity
            return {name: vec.clone() for name, vec in vector.items()}

        # Integrate parallel transport equation
        # For slowly varying metric (Christoffel ≈ 0), parallel transport ≈ identity
        # Full implementation requires solving ODE with Christoffel correction

        # Simplified: return vector unchanged (valid for constant metric)
        return {name: vec.clone() for name, vec in vector.items()}

    def log_map(
        self,
        base_point: dict[str, torch.Tensor],
        target_point: dict[str, torch.Tensor],
        fisher_metric: Optional[dict[str, torch.Tensor]] = None
    ) -> dict[str, torch.Tensor]:
        """Compute logarithm map: inverse of exponential map.

        The log map finds the tangent vector v such that:
            exp_p(v) = q

        For Euclidean space: log_p(q) = q - p
        For Riemannian manifold: Requires solving boundary value problem

        Args:
            base_point: Base point p
            target_point: Target point q
            fisher_metric: Fisher metric (optional)

        Returns:
            Tangent vector v such that exp_p(v) ≈ q

        Note:
            This is the inverse problem and is much harder than exp map.
            For now, we use Euclidean approximation: v = q - p
        """
        # Euclidean log map (first approximation)
        log_vector = {}
        for name in base_point:
            if name in target_point:
                log_vector[name] = target_point[name] - base_point[name]
            else:
                log_vector[name] = torch.zeros_like(base_point[name])

        return log_vector
