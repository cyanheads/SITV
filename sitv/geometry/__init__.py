"""
Riemannian geometry infrastructure for parameter manifolds.

This module provides:
- Fisher Information Matrix computation (metric tensor)
- Geodesic integration on parameter manifolds
- Christoffel symbols and curvature computation
- Riemannian distances and norms
- Geodesic task vector operations
"""

from sitv.geometry.config import (
    FisherApproximationType,
    GeometryConfig,
    GeodesicIntegrationConfig,
    SymmetryAnalysisConfig,
)
from sitv.geometry.geodesic import GeodesicIntegrator
from sitv.geometry.metric import FisherMetricService
from sitv.geometry.task_vector import GeodesicTaskVectorService

__all__ = [
    "FisherApproximationType",
    "GeometryConfig",
    "GeodesicIntegrationConfig",
    "SymmetryAnalysisConfig",
    "GeodesicIntegrator",
    "FisherMetricService",
    "GeodesicTaskVectorService",
]
