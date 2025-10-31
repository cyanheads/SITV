"""
Gradient analysis tools for alpha sweep experiments.

This module provides numerical gradient computation and critical point
detection for analyzing loss landscapes along task vector directions.
"""

from sitv.analysis.gradient.numerical_gradient import NumericalGradientAnalyzer
from sitv.analysis.gradient.critical_points import CriticalPointFinder, CriticalPoint

__all__ = [
    "NumericalGradientAnalyzer",
    "CriticalPointFinder",
    "CriticalPoint",
]
