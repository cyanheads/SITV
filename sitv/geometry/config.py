"""
Configuration for Riemannian geometry infrastructure.

This module provides configuration classes for geometry-related features,
including Fisher metric computation, geodesic integration, and symmetry analysis.
"""

from dataclasses import dataclass, field
from enum import Enum

# Import the YAML config loader from experiments.config
# This ensures geometry config loads from the same config.yaml
from sitv.experiments.config import _get


class FisherApproximationType(str, Enum):
    """Fisher Information Matrix approximation types.

    Attributes:
        EUCLIDEAN: Standard Euclidean metric (identity matrix, no FIM computation)
        DIAGONAL: Diagonal Fisher approximation (O(n) memory, fast)
        KFAC: Kronecker-factored approximate curvature (block-diagonal, practical)
        FULL: Full Fisher matrix (O(n²) memory, exact but expensive)
    """

    EUCLIDEAN = "euclidean"
    DIAGONAL = "fisher_diagonal"
    KFAC = "fisher_kfac"
    FULL = "fisher_full"


@dataclass
class FisherApproximationConfig:
    """Configuration for Fisher Information Matrix approximation.

    Attributes:
        sampling_strategy: How to sample data for FIM computation
            - "full": Use entire dataset (slow but exact)
            - "subset": Use random subset (fast approximation)
            - "batch": Use single batch (fastest, least accurate)
        num_samples: Number of samples to use for FIM computation (if subset)
        block_size: Block size for KFAC approximation
        eigenvalue_floor: Minimum eigenvalue to prevent singular matrices
    """

    sampling_strategy: str = field(
        default_factory=lambda: _get('geometry.fisher_approximation.sampling_strategy', 'subset')
    )
    num_samples: int = field(
        default_factory=lambda: _get('geometry.fisher_approximation.num_samples', 1000)
    )
    block_size: int = field(
        default_factory=lambda: _get('geometry.fisher_approximation.block_size', 256)
    )
    eigenvalue_floor: float = field(
        default_factory=lambda: _get('geometry.fisher_approximation.eigenvalue_floor', 1e-6)
    )


@dataclass
class GeodesicIntegrationConfig:
    """Configuration for geodesic integration on manifolds.

    Attributes:
        enabled: Whether to use geodesic interpolation (vs straight-line Euclidean)
        num_steps: Number of Runge-Kutta integration steps
        tolerance: Integration error tolerance
        step_size_control: Whether to use adaptive step size control
        max_iterations: Maximum iterations for geodesic solver
        recompute_metric_every: Recompute Fisher metric every N steps (0 = never, 1 = every step)
        metric_epsilon: Epsilon for finite difference computation of Christoffel symbols
    """

    enabled: bool = field(
        default_factory=lambda: _get('geometry.geodesic_integration.enabled', True)
    )
    num_steps: int = field(
        default_factory=lambda: _get('geometry.geodesic_integration.num_steps', 100)
    )
    tolerance: float = field(
        default_factory=lambda: _get('geometry.geodesic_integration.tolerance', 1e-6)
    )
    step_size_control: bool = field(
        default_factory=lambda: _get('geometry.geodesic_integration.step_size_control', False)
    )
    max_iterations: int = field(
        default_factory=lambda: _get('geometry.geodesic_integration.max_iterations', 1000)
    )
    recompute_metric_every: int = field(
        default_factory=lambda: _get('geometry.geodesic_integration.recompute_metric_every', 0)
    )
    metric_epsilon: float = field(
        default_factory=lambda: _get('geometry.geodesic_integration.metric_epsilon', 1e-3)
    )


@dataclass
class SymmetryAnalysisConfig:
    """Configuration for symmetry detection and quotient space analysis.

    Attributes:
        enabled: Whether to perform symmetry analysis
        detect_rotations: Test for rotation group symmetries
        detect_permutations: Test for neuron permutation symmetries
        detect_scaling: Test for layer-wise scaling symmetries
        quotient_space: Whether to work in symmetry-reduced canonical space
        symmetry_tolerance: Tolerance for detecting symmetries (L(R·θ) ≈ L(θ))
    """

    enabled: bool = field(
        default_factory=lambda: _get('geometry.symmetry_analysis.enabled', False)
    )
    detect_rotations: bool = field(
        default_factory=lambda: _get('geometry.symmetry_analysis.detect_rotations', True)
    )
    detect_permutations: bool = field(
        default_factory=lambda: _get('geometry.symmetry_analysis.detect_permutations', True)
    )
    detect_scaling: bool = field(
        default_factory=lambda: _get('geometry.symmetry_analysis.detect_scaling', True)
    )
    quotient_space: bool = field(
        default_factory=lambda: _get('geometry.symmetry_analysis.quotient_space', False)
    )
    symmetry_tolerance: float = field(
        default_factory=lambda: _get('geometry.symmetry_analysis.symmetry_tolerance', 0.01)
    )


@dataclass
class CurvatureAnalysisConfig:
    """Configuration for Riemannian curvature computation.

    Attributes:
        enabled: Whether to compute curvature tensors
        compute_sectional: Compute sectional curvature K(X,Y)
        compute_ricci: Compute Ricci curvature tensor
        compute_scalar: Compute scalar curvature
        num_tangent_samples: Number of random tangent vectors for curvature estimation
    """

    enabled: bool = field(
        default_factory=lambda: _get('geometry.curvature_analysis.enabled', False)
    )
    compute_sectional: bool = field(
        default_factory=lambda: _get('geometry.curvature_analysis.compute_sectional', True)
    )
    compute_ricci: bool = field(
        default_factory=lambda: _get('geometry.curvature_analysis.compute_ricci', False)
    )
    compute_scalar: bool = field(
        default_factory=lambda: _get('geometry.curvature_analysis.compute_scalar', False)
    )
    num_tangent_samples: int = field(
        default_factory=lambda: _get('geometry.curvature_analysis.num_tangent_samples', 10)
    )


@dataclass
class ChristoffelComputationConfig:
    """Configuration for Christoffel symbol computation.

    Attributes:
        skip_vision_tower: Skip parameters with 'vision_tower' in name
        skip_frozen: Skip parameters where requires_grad=False
        num_samples: Number of data samples to use for Fisher computation
        parameter_sample_fraction: Fraction of parameters to compute (0.0-1.0, 1.0=all)
        max_parameters: Maximum number of parameters to process (None=unlimited)
    """

    skip_vision_tower: bool = field(
        default_factory=lambda: _get('geometry.christoffel_computation.skip_vision_tower', True)
    )
    skip_frozen: bool = field(
        default_factory=lambda: _get('geometry.christoffel_computation.skip_frozen', True)
    )
    num_samples: int = field(
        default_factory=lambda: _get('geometry.christoffel_computation.num_samples', 20)
    )
    parameter_sample_fraction: float = field(
        default_factory=lambda: _get('geometry.christoffel_computation.parameter_sample_fraction', 1.0)
    )
    max_parameters: int | None = field(
        default_factory=lambda: _get('geometry.christoffel_computation.max_parameters', None)
    )


@dataclass
class GeometryConfig:
    """Complete geometry configuration.

    This configuration controls the Riemannian geometry infrastructure,
    including Fisher metric computation, geodesic integration, and
    symmetry/curvature analysis.

    Attributes:
        enabled: Master switch for Riemannian geometry features
        metric_type: Type of metric tensor to use
        fisher_approximation: Fisher matrix approximation settings
        geodesic_integration: Geodesic integration settings
        christoffel_computation: Christoffel symbol computation settings
        symmetry_analysis: Symmetry detection settings
        curvature_analysis: Curvature computation settings
        cache_metric: Whether to cache computed Fisher metrics
        parallel_transport: Whether to use parallel transport for task vectors
    """

    enabled: bool = field(
        default_factory=lambda: _get('geometry.enabled', False)
    )
    metric_type: FisherApproximationType = field(
        default_factory=lambda: FisherApproximationType(
            _get('geometry.metric_type', 'euclidean')
        )
    )
    cache_metric: bool = field(
        default_factory=lambda: _get('geometry.cache_metric', True)
    )
    parallel_transport: bool = field(
        default_factory=lambda: _get('geometry.parallel_transport', False)
    )

    # Sub-configurations
    fisher_approximation: FisherApproximationConfig = field(
        default_factory=FisherApproximationConfig
    )
    geodesic_integration: GeodesicIntegrationConfig = field(
        default_factory=GeodesicIntegrationConfig
    )
    christoffel_computation: ChristoffelComputationConfig = field(
        default_factory=ChristoffelComputationConfig
    )
    symmetry_analysis: SymmetryAnalysisConfig = field(
        default_factory=SymmetryAnalysisConfig
    )
    curvature_analysis: CurvatureAnalysisConfig = field(
        default_factory=CurvatureAnalysisConfig
    )

    @property
    def use_riemannian(self) -> bool:
        """Whether to use Riemannian geometry (vs Euclidean)."""
        return self.enabled and self.metric_type != FisherApproximationType.EUCLIDEAN

    @property
    def use_geodesics(self) -> bool:
        """Whether to use geodesic interpolation."""
        return self.use_riemannian and self.geodesic_integration.enabled

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "enabled": self.enabled,
            "metric_type": self.metric_type.value,
            "cache_metric": self.cache_metric,
            "parallel_transport": self.parallel_transport,
            "fisher_approximation": {
                "sampling_strategy": self.fisher_approximation.sampling_strategy,
                "num_samples": self.fisher_approximation.num_samples,
                "block_size": self.fisher_approximation.block_size,
                "eigenvalue_floor": self.fisher_approximation.eigenvalue_floor,
            },
            "geodesic_integration": {
                "enabled": self.geodesic_integration.enabled,
                "num_steps": self.geodesic_integration.num_steps,
                "tolerance": self.geodesic_integration.tolerance,
                "step_size_control": self.geodesic_integration.step_size_control,
            },
            "christoffel_computation": {
                "skip_vision_tower": self.christoffel_computation.skip_vision_tower,
                "skip_frozen": self.christoffel_computation.skip_frozen,
                "num_samples": self.christoffel_computation.num_samples,
                "parameter_sample_fraction": self.christoffel_computation.parameter_sample_fraction,
                "max_parameters": self.christoffel_computation.max_parameters,
            },
            "symmetry_analysis": {
                "enabled": self.symmetry_analysis.enabled,
                "detect_rotations": self.symmetry_analysis.detect_rotations,
                "detect_permutations": self.symmetry_analysis.detect_permutations,
                "quotient_space": self.symmetry_analysis.quotient_space,
            },
            "curvature_analysis": {
                "enabled": self.curvature_analysis.enabled,
                "compute_sectional": self.curvature_analysis.compute_sectional,
                "compute_ricci": self.curvature_analysis.compute_ricci,
            },
        }
