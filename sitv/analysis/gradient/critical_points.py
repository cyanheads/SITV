"""
Critical point detection for alpha sweep results.

This module provides tools for finding and classifying critical points
(minima, maxima, inflection points) in the loss landscape.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sitv.data.models import AlphaSweepResult
from sitv.analysis.gradient.numerical_gradient import NumericalGradientAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class CriticalPoint:
    """Represents a critical point in the loss landscape.

    Attributes:
        alpha: Alpha value at critical point
        loss: Loss value at critical point
        point_type: Type of critical point ('minimum', 'maximum', 'inflection', 'saddle')
        gradient: First derivative at this point
        curvature: Second derivative at this point
        significance: How significant this point is (0-1 scale)
    """
    alpha: float
    loss: float
    point_type: str
    gradient: float
    curvature: float
    significance: float = 1.0


class CriticalPointFinder:
    """Find and classify critical points in loss landscapes.

    This class identifies local minima, maxima, and inflection points
    by analyzing gradients and curvatures.

    Attributes:
        gradient_analyzer: Numerical gradient analyzer
        gradient_threshold: Threshold for "zero" gradient
        curvature_threshold: Minimum curvature for classification
    """

    def __init__(
        self,
        gradient_threshold: float = 0.01,
        curvature_threshold: float = 0.001,
        smooth_sigma: float = 0.5
    ):
        """Initialize critical point finder.

        Args:
            gradient_threshold: Threshold for zero gradient detection
            curvature_threshold: Minimum curvature for classification
            smooth_sigma: Smoothing parameter for gradient computation
        """
        self.gradient_threshold = gradient_threshold
        self.curvature_threshold = curvature_threshold
        self.gradient_analyzer = NumericalGradientAnalyzer(smooth_sigma=smooth_sigma)

    def find_all_critical_points(
        self,
        results: List[AlphaSweepResult]
    ) -> Dict[str, List[CriticalPoint]]:
        """Find all critical points in the loss landscape.

        Args:
            results: List of alpha sweep results

        Returns:
            Dictionary with keys:
                - 'minima': List of local minima
                - 'maxima': List of local maxima
                - 'inflection': List of inflection points
                - 'all': All critical points combined

        Example:
            >>> finder = CriticalPointFinder()
            >>> critical_points = finder.find_all_critical_points(results)
            >>> print(f"Found {len(critical_points['minima'])} local minima")
        """
        if len(results) < 5:
            logger.warning("Not enough results for critical point analysis (need at least 5)")
            return {'minima': [], 'maxima': [], 'inflection': [], 'all': []}

        # Compute gradients and curvatures
        alphas_grad, gradients = self.gradient_analyzer.compute_gradients(results)
        alphas_curv, curvatures = self.gradient_analyzer.compute_second_derivative(results)

        # Find minima and maxima (where gradient ≈ 0)
        minima = self.find_minima(results, alphas_grad, gradients, alphas_curv, curvatures)
        maxima = self.find_maxima(results, alphas_grad, gradients, alphas_curv, curvatures)

        # Find inflection points (where curvature changes sign)
        inflection = self.find_inflection_points(results, alphas_curv, curvatures)

        all_points = minima + maxima + inflection

        return {
            'minima': minima,
            'maxima': maxima,
            'inflection': inflection,
            'all': all_points
        }

    def find_minima(
        self,
        results: List[AlphaSweepResult],
        alphas_grad: np.ndarray,
        gradients: np.ndarray,
        alphas_curv: np.ndarray,
        curvatures: np.ndarray
    ) -> List[CriticalPoint]:
        """Find local minima (gradient ≈ 0, curvature > 0).

        Args:
            results: Alpha sweep results
            alphas_grad: Alpha values for gradient
            gradients: First derivatives
            alphas_curv: Alpha values for curvature
            curvatures: Second derivatives

        Returns:
            List of CriticalPoint objects representing minima
        """
        minima = []

        # Find zero-gradient points
        zero_grad_indices = self._find_zero_crossings(gradients)

        for idx in zero_grad_indices:
            if idx == 0 or idx >= len(alphas_grad) - 1:
                continue  # Skip boundary points

            alpha = alphas_grad[idx]

            # Interpolate curvature at this alpha
            curv = self._interpolate_curvature(alpha, alphas_curv, curvatures)

            # Check if it's a minimum (positive curvature)
            if curv > self.curvature_threshold:
                # Get corresponding loss value
                loss = self._get_loss_at_alpha(results, alpha)

                # Compute significance (larger curvature = more significant minimum)
                significance = min(1.0, abs(curv) / 1.0)  # Normalize to 0-1

                minima.append(CriticalPoint(
                    alpha=alpha,
                    loss=loss,
                    point_type='minimum',
                    gradient=gradients[idx],
                    curvature=curv,
                    significance=significance
                ))

        return minima

    def find_maxima(
        self,
        results: List[AlphaSweepResult],
        alphas_grad: np.ndarray,
        gradients: np.ndarray,
        alphas_curv: np.ndarray,
        curvatures: np.ndarray
    ) -> List[CriticalPoint]:
        """Find local maxima (gradient ≈ 0, curvature < 0).

        Args:
            results: Alpha sweep results
            alphas_grad: Alpha values for gradient
            gradients: First derivatives
            alphas_curv: Alpha values for curvature
            curvatures: Second derivatives

        Returns:
            List of CriticalPoint objects representing maxima
        """
        maxima = []

        # Find zero-gradient points
        zero_grad_indices = self._find_zero_crossings(gradients)

        for idx in zero_grad_indices:
            if idx == 0 or idx >= len(alphas_grad) - 1:
                continue  # Skip boundary points

            alpha = alphas_grad[idx]

            # Interpolate curvature at this alpha
            curv = self._interpolate_curvature(alpha, alphas_curv, curvatures)

            # Check if it's a maximum (negative curvature)
            if curv < -self.curvature_threshold:
                # Get corresponding loss value
                loss = self._get_loss_at_alpha(results, alpha)

                # Compute significance
                significance = min(1.0, abs(curv) / 1.0)

                maxima.append(CriticalPoint(
                    alpha=alpha,
                    loss=loss,
                    point_type='maximum',
                    gradient=gradients[idx],
                    curvature=curv,
                    significance=significance
                ))

        return maxima

    def find_inflection_points(
        self,
        results: List[AlphaSweepResult],
        alphas_curv: np.ndarray,
        curvatures: np.ndarray
    ) -> List[CriticalPoint]:
        """Find inflection points (where curvature changes sign).

        Args:
            results: Alpha sweep results
            alphas_curv: Alpha values for curvature
            curvatures: Second derivatives

        Returns:
            List of CriticalPoint objects representing inflection points
        """
        inflection = []

        # Find where curvature changes sign
        sign_changes = np.diff(np.sign(curvatures))
        inflection_indices = np.where(sign_changes != 0)[0]

        for idx in inflection_indices:
            if idx >= len(alphas_curv) - 1:
                continue

            # Interpolate alpha where curvature crosses zero
            alpha = self._interpolate_zero_crossing(
                alphas_curv[idx],
                alphas_curv[idx + 1],
                curvatures[idx],
                curvatures[idx + 1]
            )

            # Get loss at this alpha
            loss = self._get_loss_at_alpha(results, alpha)

            # Get gradient at this point
            grad = self._get_gradient_at_alpha(results, alpha)

            # Compute significance (larger gradient = more significant inflection)
            significance = min(1.0, abs(grad) / 1.0)

            inflection.append(CriticalPoint(
                alpha=alpha,
                loss=loss,
                point_type='inflection',
                gradient=grad,
                curvature=0.0,  # By definition
                significance=significance
            ))

        return inflection

    def _find_zero_crossings(
        self,
        values: np.ndarray
    ) -> List[int]:
        """Find indices where values cross zero.

        Args:
            values: Array of values

        Returns:
            List of indices near zero-crossings
        """
        indices = []

        # Find sign changes
        signs = np.sign(values)
        sign_changes = np.diff(signs)
        crossing_indices = np.where(sign_changes != 0)[0]

        for idx in crossing_indices:
            indices.append(idx)

        # Also include points very close to zero
        near_zero = np.where(np.abs(values) < self.gradient_threshold)[0]
        indices.extend(near_zero)

        # Remove duplicates and sort
        indices = sorted(set(indices))

        return indices

    def _interpolate_curvature(
        self,
        alpha: float,
        alphas_curv: np.ndarray,
        curvatures: np.ndarray
    ) -> float:
        """Interpolate curvature at a specific alpha.

        Args:
            alpha: Target alpha value
            alphas_curv: Alpha values where curvature is known
            curvatures: Curvature values

        Returns:
            Interpolated curvature
        """
        return float(np.interp(alpha, alphas_curv, curvatures))

    def _interpolate_zero_crossing(
        self,
        x1: float,
        x2: float,
        y1: float,
        y2: float
    ) -> float:
        """Interpolate x where y crosses zero between two points.

        Args:
            x1, x2: X coordinates
            y1, y2: Y coordinates

        Returns:
            Interpolated x where y ≈ 0
        """
        if abs(y2 - y1) < 1e-10:
            return (x1 + x2) / 2

        # Linear interpolation
        return x1 - y1 * (x2 - x1) / (y2 - y1)

    def _get_loss_at_alpha(
        self,
        results: List[AlphaSweepResult],
        alpha: float
    ) -> float:
        """Get loss value at a specific alpha (interpolated if needed).

        Args:
            results: Alpha sweep results
            alpha: Target alpha value

        Returns:
            Loss value at alpha
        """
        sorted_results = sorted(results, key=lambda r: r.alpha)
        alphas = np.array([r.alpha for r in sorted_results])
        losses = np.array([r.loss for r in sorted_results])

        return float(np.interp(alpha, alphas, losses))

    def _get_gradient_at_alpha(
        self,
        results: List[AlphaSweepResult],
        alpha: float
    ) -> float:
        """Get gradient at a specific alpha.

        Args:
            results: Alpha sweep results
            alpha: Target alpha value

        Returns:
            Gradient at alpha
        """
        alphas_grad, gradients = self.gradient_analyzer.compute_gradients(results)
        return float(np.interp(alpha, alphas_grad, gradients))

    def classify_critical_points(
        self,
        results: List[AlphaSweepResult]
    ) -> Dict:
        """Full classification of all critical points with analysis.

        Args:
            results: Alpha sweep results

        Returns:
            Dictionary with detailed analysis including:
                - Critical points by type
                - Global minimum/maximum
                - Most significant points
                - Summary statistics

        Example:
            >>> finder = CriticalPointFinder()
            >>> analysis = finder.classify_critical_points(results)
            >>> global_min = analysis['global_minimum']
            >>> print(f"Global minimum at α={global_min.alpha:.3f}, L={global_min.loss:.4f}")
        """
        critical_points = self.find_all_critical_points(results)

        analysis = {
            **critical_points,
            'summary': {
                'num_minima': len(critical_points['minima']),
                'num_maxima': len(critical_points['maxima']),
                'num_inflection': len(critical_points['inflection']),
                'total_critical_points': len(critical_points['all'])
            }
        }

        # Find global minimum and maximum
        if critical_points['minima']:
            analysis['global_minimum'] = min(critical_points['minima'], key=lambda cp: cp.loss)  # type: ignore[type-var,arg-type,attr-defined,assignment]
        else:
            # If no critical minima, use boundary
            sorted_results = sorted(results, key=lambda r: r.loss)
            best_result = sorted_results[0]
            analysis['global_minimum'] = CriticalPoint(  # type: ignore[assignment]
                alpha=best_result.alpha,
                loss=best_result.loss,
                point_type='boundary_minimum',
                gradient=0.0,
                curvature=0.0,
                significance=1.0
            )

        if critical_points['maxima']:
            analysis['global_maximum'] = max(critical_points['maxima'], key=lambda cp: cp.loss)  # type: ignore[type-var,arg-type,attr-defined]

        # Most significant points
        if critical_points['all']:
            analysis['most_significant'] = sorted(
                critical_points['all'],
                key=lambda cp: cp.significance,
                reverse=True
            )[:5]

        return analysis
