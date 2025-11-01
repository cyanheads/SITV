"""
Numerical gradient computation for alpha sweep results.

This module provides tools for computing numerical derivatives of the
loss landscape along the task vector direction.
"""

import logging
from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d  # type: ignore[import-untyped]

from sitv.data.models import AlphaSweepResult

logger = logging.getLogger(__name__)


class NumericalGradientAnalyzer:
    """Compute numerical gradients from alpha sweep results.

    This class computes first and second derivatives of the loss function
    with respect to alpha using finite differences.

    Attributes:
        smooth_sigma: Gaussian smoothing parameter (0 = no smoothing)
        method: Finite difference method ('central', 'forward', 'backward')
    """

    def __init__(self, smooth_sigma: float = 0.5, method: str = "central"):
        """Initialize gradient analyzer.

        Args:
            smooth_sigma: Standard deviation for Gaussian smoothing (0 = no smoothing)
            method: Finite difference method ('central' recommended)
        """
        self.smooth_sigma = smooth_sigma
        self.method = method

        if method not in ["central", "forward", "backward"]:
            raise ValueError(
                f"Invalid method: {method}. Must be 'central', 'forward', or 'backward'"
            )

    def compute_gradients(
        self, results: list[AlphaSweepResult], smooth: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute first derivative dL/dα.

        Args:
            results: List of alpha sweep results (sorted by alpha)
            smooth: Whether to apply Gaussian smoothing

        Returns:
            Tuple of (alphas, gradients)
                - alphas: Alpha values where gradients are computed
                - gradients: dL/dα at each alpha

        Example:
            >>> analyzer = NumericalGradientAnalyzer()
            >>> alphas, grads = analyzer.compute_gradients(results)
            >>> print(f"Gradient at α=1.0: {grads[closest_idx_to_1]}")
        """
        # Sort results by alpha
        sorted_results = sorted(results, key=lambda r: r.alpha)
        alphas = np.array([r.alpha for r in sorted_results])
        losses = np.array([r.loss for r in sorted_results])

        # Apply smoothing if requested
        if smooth and self.smooth_sigma > 0:
            losses = gaussian_filter1d(losses, self.smooth_sigma)

        # Compute gradients using finite differences
        if self.method == "central":
            gradients = self._central_difference(alphas, losses)
        elif self.method == "forward":
            gradients = self._forward_difference(alphas, losses)
        else:  # backward
            gradients = self._backward_difference(alphas, losses)

        return alphas, gradients

    def compute_second_derivative(
        self, results: list[AlphaSweepResult], smooth: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute second derivative d²L/dα² (curvature).

        Args:
            results: List of alpha sweep results (sorted by alpha)
            smooth: Whether to apply Gaussian smoothing

        Returns:
            Tuple of (alphas, second_derivatives)
                - alphas: Alpha values where curvatures are computed
                - second_derivatives: d²L/dα² at each alpha

        Example:
            >>> analyzer = NumericalGradientAnalyzer()
            >>> alphas, curvs = analyzer.compute_second_derivative(results)
            >>> positive_curvature_mask = curvs > 0  # Convex regions
        """
        # Sort results by alpha
        sorted_results = sorted(results, key=lambda r: r.alpha)
        alphas = np.array([r.alpha for r in sorted_results])
        losses = np.array([r.loss for r in sorted_results])

        # Apply smoothing if requested (more aggressive for second derivative)
        if smooth and self.smooth_sigma > 0:
            losses = gaussian_filter1d(losses, self.smooth_sigma * 1.5)

        # Compute second derivative
        second_deriv = self._second_order_central_difference(alphas, losses)

        # Return corresponding alphas (middle points)
        return alphas[1:-1], second_deriv

    def _central_difference(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient using central differences.

        More accurate than forward/backward for interior points.

        Args:
            x: Independent variable (alpha)
            y: Dependent variable (loss)

        Returns:
            Array of gradients (same length as x)
        """
        grad = np.zeros_like(y)

        # Interior points: central difference
        for i in range(1, len(y) - 1):
            h_forward = x[i + 1] - x[i]
            h_backward = x[i] - x[i - 1]

            # Weighted central difference for non-uniform spacing
            grad[i] = ((y[i + 1] - y[i]) / h_forward + (y[i] - y[i - 1]) / h_backward) / 2

        # Boundary points: forward/backward difference
        grad[0] = (y[1] - y[0]) / (x[1] - x[0])
        grad[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

        return grad

    def _forward_difference(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient using forward differences.

        Args:
            x: Independent variable (alpha)
            y: Dependent variable (loss)

        Returns:
            Array of gradients (same length as x)
        """
        grad = np.zeros_like(y)

        for i in range(len(y) - 1):
            h = x[i + 1] - x[i]
            grad[i] = (y[i + 1] - y[i]) / h

        # Last point uses backward difference
        grad[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

        return grad

    def _backward_difference(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute gradient using backward differences.

        Args:
            x: Independent variable (alpha)
            y: Dependent variable (loss)

        Returns:
            Array of gradients (same length as x)
        """
        grad = np.zeros_like(y)

        # First point uses forward difference
        grad[0] = (y[1] - y[0]) / (x[1] - x[0])

        for i in range(1, len(y)):
            h = x[i] - x[i - 1]
            grad[i] = (y[i] - y[i - 1]) / h

        return grad

    def _second_order_central_difference(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute second derivative using central differences.

        Args:
            x: Independent variable (alpha)
            y: Dependent variable (loss)

        Returns:
            Array of second derivatives (length = len(x) - 2)
        """
        second_deriv = np.zeros(len(y) - 2)

        for i in range(1, len(y) - 1):
            h_forward = x[i + 1] - x[i]
            h_backward = x[i] - x[i - 1]
            h_avg = (h_forward + h_backward) / 2

            # Second derivative approximation
            second_deriv[i - 1] = (
                (y[i + 1] - y[i]) / h_forward - (y[i] - y[i - 1]) / h_backward
            ) / h_avg

        return second_deriv

    def estimate_noise_level(self, results: list[AlphaSweepResult]) -> float:
        """Estimate noise level in the loss measurements.

        This helps determine appropriate smoothing parameters.

        Args:
            results: List of alpha sweep results

        Returns:
            Estimated standard deviation of noise
        """
        if len(results) < 3:
            return 0.0

        sorted_results = sorted(results, key=lambda r: r.alpha)
        losses = np.array([r.loss for r in sorted_results])

        # Compute second differences
        second_diff = np.diff(losses, n=2)

        # Estimate noise from second differences
        # (assuming smooth underlying function)
        noise_std = np.std(second_diff) / np.sqrt(6)

        return float(noise_std)
