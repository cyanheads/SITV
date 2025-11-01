"""
Riemannian geometry infrastructure for parameter manifolds.

This module provides:
- Fisher Information Matrix computation (metric tensor)
- Geodesic integration on parameter manifolds
- Christoffel symbols and curvature computation
- Riemannian distances and norms
- Geodesic task vector operations
- Curvature analysis (sectional, Ricci, scalar curvature)
- Symmetry detection and quotient space projection
"""

from sitv.geometry.config import (
    CurvatureAnalysisConfig,
    FisherApproximationType,
    GeometryConfig,
    GeodesicIntegrationConfig,
    SymmetryAnalysisConfig,
)
from sitv.geometry.curvature import CurvatureAnalyzer
from sitv.geometry.geodesic import GeodesicIntegrator
from sitv.geometry.metric import FisherMetricService
from sitv.geometry.symmetry import SymmetryAnalyzer
from sitv.geometry.task_vector import GeodesicTaskVectorService

__all__ = [
    "CurvatureAnalysisConfig",
    "CurvatureAnalyzer",
    "FisherApproximationType",
    "GeometryConfig",
    "GeodesicIntegrationConfig",
    "GeodesicIntegrator",
    "FisherMetricService",
    "GeodesicTaskVectorService",
    "SymmetryAnalysisConfig",
    "SymmetryAnalyzer",
]
