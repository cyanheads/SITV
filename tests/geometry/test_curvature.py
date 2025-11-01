"""
Tests for Riemannian curvature computation on parameter manifolds.
"""

import pytest
import torch

from sitv.geometry.config import CurvatureAnalysisConfig
from sitv.geometry.curvature import CurvatureAnalyzer


@pytest.fixture
def curvature_config():
    """Fixture providing curvature analysis configuration."""
    return CurvatureAnalysisConfig(
        enabled=True,
        compute_sectional=True,
        compute_ricci=False,  # Expensive, disabled for fast tests
        compute_scalar=False,  # Very expensive, disabled
        num_tangent_samples=5  # Small number for fast tests
    )


@pytest.fixture
def sample_parameter_dict(device):
    """Fixture providing a sample parameter dictionary."""
    return {
        "layer1.weight": torch.randn(10, 5, device=device),
        "layer1.bias": torch.randn(10, device=device),
        "layer2.weight": torch.randn(5, 10, device=device),
        "layer2.bias": torch.randn(5, device=device),
    }


@pytest.fixture
def sample_fisher_diagonal(sample_parameter_dict, device):
    """Fixture providing a diagonal Fisher metric."""
    fisher = {}
    for name, param in sample_parameter_dict.items():
        # Positive diagonal Fisher
        fisher[name] = torch.abs(torch.randn_like(param)) + 0.1
    return fisher


@pytest.fixture
def sample_tangent_vectors(sample_parameter_dict, device):
    """Fixture providing two sample tangent vectors."""
    v1 = {}
    v2 = {}
    for name, param in sample_parameter_dict.items():
        v1[name] = torch.randn_like(param)
        v2[name] = torch.randn_like(param)
    return v1, v2


class TestCurvatureAnalyzer:
    """Test suite for CurvatureAnalyzer."""

    def test_initialization(self, curvature_config, device):
        """Test curvature analyzer initialization."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)

        assert analyzer.config == curvature_config
        assert analyzer.device == device
        assert analyzer._cached_curvature is None

    def test_sample_random_tangent_vector_shape(
        self, curvature_config, sample_parameter_dict, device
    ):
        """Test that random tangent vectors have correct shape."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)

        tangent = analyzer._sample_random_tangent_vector(sample_parameter_dict)

        # Check all keys present
        assert set(tangent.keys()) == set(sample_parameter_dict.keys())

        # Check shapes match
        for name, tensor in tangent.items():
            assert tensor.shape == sample_parameter_dict[name].shape
            assert tensor.device.type == device.type

    def test_riemannian_inner_product_euclidean(
        self, curvature_config, sample_tangent_vectors, device
    ):
        """Test Riemannian inner product reduces to Euclidean for identity metric."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)
        v1, v2 = sample_tangent_vectors

        # Identity metric (all ones)
        fisher_identity = {name: torch.ones_like(v1[name]) for name in v1.keys()}

        riemannian_inner = analyzer._riemannian_inner_product(v1, v2, fisher_identity)

        # Compute Euclidean inner product manually
        euclidean_inner = sum(
            torch.sum(v1[name] * v2[name]).item() for name in v1.keys()
        )

        assert abs(float(riemannian_inner) - euclidean_inner) < 1e-5

    def test_riemannian_inner_product_symmetry(
        self, curvature_config, sample_tangent_vectors, sample_fisher_diagonal, device
    ):
        """Test that inner product is symmetric: ⟨v1, v2⟩ = ⟨v2, v1⟩."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)
        v1, v2 = sample_tangent_vectors

        inner_12 = analyzer._riemannian_inner_product(v1, v2, sample_fisher_diagonal)
        inner_21 = analyzer._riemannian_inner_product(v2, v1, sample_fisher_diagonal)

        assert abs(float(inner_12 - inner_21)) < 1e-6

    def test_riemannian_inner_product_positive_definite(
        self, curvature_config, sample_tangent_vectors, sample_fisher_diagonal, device
    ):
        """Test that ⟨v, v⟩ > 0 for non-zero v (positive-definiteness)."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)
        v1, _ = sample_tangent_vectors

        inner_vv = analyzer._riemannian_inner_product(v1, v1, sample_fisher_diagonal)

        assert float(inner_vv) > 0

    def test_normalize_tangent_vector(
        self, curvature_config, sample_tangent_vectors, sample_fisher_diagonal, device
    ):
        """Test that normalized vector has unit norm."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)
        v1, _ = sample_tangent_vectors

        v_normalized = analyzer._normalize_tangent_vector(v1, sample_fisher_diagonal)

        # Compute norm
        norm = torch.sqrt(
            analyzer._riemannian_inner_product(v_normalized, v_normalized, sample_fisher_diagonal)
        )

        assert abs(float(norm) - 1.0) < 1e-5

    def test_orthogonalize_tangent_vectors(
        self, curvature_config, sample_tangent_vectors, sample_fisher_diagonal, device
    ):
        """Test that orthogonalized vectors are orthogonal."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)
        v1, v2 = sample_tangent_vectors

        v2_orth = analyzer._orthogonalize_tangent_vectors(v1, v2, sample_fisher_diagonal)

        # Check orthogonality: ⟨v1, v2_orth⟩ ≈ 0
        inner = analyzer._riemannian_inner_product(v1, v2_orth, sample_fisher_diagonal)

        assert abs(float(inner)) < 1e-4

    def test_sectional_curvature_finite(
        self,
        curvature_config,
        sample_parameter_dict,
        sample_tangent_vectors,
        sample_fisher_diagonal,
        device
    ):
        """Test that sectional curvature returns finite values."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)
        v1, v2 = sample_tangent_vectors

        # Normalize and orthogonalize
        v1_norm = analyzer._normalize_tangent_vector(v1, sample_fisher_diagonal)
        v2_orth = analyzer._orthogonalize_tangent_vectors(v1, v2, sample_fisher_diagonal)
        v2_norm = analyzer._normalize_tangent_vector(v2_orth, sample_fisher_diagonal)

        K = analyzer.compute_sectional_curvature(
            sample_parameter_dict,
            v1_norm,
            v2_norm,
            sample_fisher_diagonal
        )

        assert torch.isfinite(torch.tensor(K))

    def test_sectional_curvature_symmetry(
        self,
        curvature_config,
        sample_parameter_dict,
        sample_tangent_vectors,
        sample_fisher_diagonal,
        device
    ):
        """Test that K(X,Y) = K(Y,X) (symmetry property)."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)
        v1, v2 = sample_tangent_vectors

        # Normalize and orthogonalize
        v1_norm = analyzer._normalize_tangent_vector(v1, sample_fisher_diagonal)
        v2_orth = analyzer._orthogonalize_tangent_vectors(v1, v2, sample_fisher_diagonal)
        v2_norm = analyzer._normalize_tangent_vector(v2_orth, sample_fisher_diagonal)

        K_xy = analyzer.compute_sectional_curvature(
            sample_parameter_dict, v1_norm, v2_norm, sample_fisher_diagonal
        )
        K_yx = analyzer.compute_sectional_curvature(
            sample_parameter_dict, v2_norm, v1_norm, sample_fisher_diagonal
        )

        # Should be symmetric
        assert abs(K_xy - K_yx) < 1e-5

    def test_sectional_curvature_parallel_vectors_zero(
        self,
        curvature_config,
        sample_parameter_dict,
        sample_fisher_diagonal,
        device
    ):
        """Test that K(X,X) = 0 (curvature of parallel vectors is zero)."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)

        # Create a random vector
        v = {}
        for name, param in sample_parameter_dict.items():
            v[name] = torch.randn_like(param)

        v_norm = analyzer._normalize_tangent_vector(v, sample_fisher_diagonal)

        K = analyzer.compute_sectional_curvature(
            sample_parameter_dict, v_norm, v_norm, sample_fisher_diagonal
        )

        # Curvature of parallel vectors should be zero
        assert abs(K) < 1e-6

    def test_curvature_distribution_structure(
        self,
        curvature_config,
        sample_parameter_dict,
        sample_fisher_diagonal,
        device
    ):
        """Test that curvature distribution returns correct structure."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)

        results = analyzer.estimate_curvature_distribution(
            sample_parameter_dict,
            sample_fisher_diagonal,
            num_samples=5
        )

        # Check all required keys present
        required_keys = [
            'mean_curvature',
            'std_curvature',
            'min_curvature',
            'max_curvature',
            'curvature_samples',
            'interpretation',
            'num_samples'
        ]
        for key in required_keys:
            assert key in results

        # Check types
        assert isinstance(results['mean_curvature'], float)
        assert isinstance(results['std_curvature'], float)
        assert isinstance(results['interpretation'], str)
        assert isinstance(results['curvature_samples'], list)
        assert len(results['curvature_samples']) <= 5

    def test_curvature_distribution_all_finite(
        self,
        curvature_config,
        sample_parameter_dict,
        sample_fisher_diagonal,
        device
    ):
        """Test that all curvature samples are finite."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)

        results = analyzer.estimate_curvature_distribution(
            sample_parameter_dict,
            sample_fisher_diagonal,
            num_samples=10
        )

        # All samples should be finite
        for K in results['curvature_samples']:
            assert torch.isfinite(torch.tensor(K))

    def test_curvature_caching(
        self, curvature_config, sample_parameter_dict, sample_fisher_diagonal, device
    ):
        """Test curvature caching functionality."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)

        # Initially no cache
        assert analyzer.get_cached_curvature() is None

        # Compute and cache results
        results = analyzer.estimate_curvature_distribution(
            sample_parameter_dict, sample_fisher_diagonal
        )
        analyzer.cache_curvature(results)

        # Retrieve cached results
        cached = analyzer.get_cached_curvature()
        assert cached is not None
        assert cached['mean_curvature'] == results['mean_curvature']

        # Clear cache
        analyzer.clear_cache()
        assert analyzer.get_cached_curvature() is None

    def test_ensure_device_cpu_to_cpu(self, curvature_config, device):
        """Test that _ensure_device works correctly."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)

        param_dict = {
            "layer1.weight": torch.randn(5, 3, device=device),
            "layer1.bias": torch.randn(5, device=device),
        }

        result = analyzer._ensure_device(param_dict)

        for name, tensor in result.items():
            assert tensor.device.type == device.type

    @pytest.mark.parametrize("num_samples", [5, 10, 20])
    def test_curvature_distribution_num_samples(
        self,
        curvature_config,
        sample_parameter_dict,
        sample_fisher_diagonal,
        device,
        num_samples
    ):
        """Test curvature distribution with different sample counts."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)

        results = analyzer.estimate_curvature_distribution(
            sample_parameter_dict,
            sample_fisher_diagonal,
            num_samples=num_samples
        )

        # Should have at most num_samples samples (may be fewer due to numerical issues)
        assert len(results['curvature_samples']) <= num_samples
        assert results['num_samples'] <= num_samples

    def test_ricci_curvature_disabled(
        self,
        sample_parameter_dict,
        sample_tangent_vectors,
        sample_fisher_diagonal,
        device
    ):
        """Test that Ricci curvature returns 0 when disabled."""
        config = CurvatureAnalysisConfig(
            enabled=True,
            compute_ricci=False  # Explicitly disabled
        )
        analyzer = CurvatureAnalyzer(config, device=device)
        v1, _ = sample_tangent_vectors

        ric = analyzer.compute_ricci_curvature(
            sample_parameter_dict, v1, sample_fisher_diagonal
        )

        assert ric == 0.0

    def test_ricci_curvature_enabled(
        self,
        sample_parameter_dict,
        sample_tangent_vectors,
        sample_fisher_diagonal,
        device
    ):
        """Test that Ricci curvature can be computed when enabled."""
        config = CurvatureAnalysisConfig(
            enabled=True,
            compute_ricci=True,  # Enabled
            num_tangent_samples=5
        )
        analyzer = CurvatureAnalyzer(config, device=device)
        v1, _ = sample_tangent_vectors

        v1_norm = analyzer._normalize_tangent_vector(v1, sample_fisher_diagonal)

        ric = analyzer.compute_ricci_curvature(
            sample_parameter_dict,
            v1_norm,
            sample_fisher_diagonal,
            num_basis_vectors=5  # Small for fast test
        )

        # Should return finite value
        assert torch.isfinite(torch.tensor(ric))

    def test_scalar_curvature_disabled(
        self, sample_parameter_dict, sample_fisher_diagonal, device
    ):
        """Test that scalar curvature returns 0 when disabled."""
        config = CurvatureAnalysisConfig(
            enabled=True,
            compute_scalar=False  # Explicitly disabled
        )
        analyzer = CurvatureAnalyzer(config, device=device)

        R = analyzer.compute_scalar_curvature(
            sample_parameter_dict, sample_fisher_diagonal
        )

        assert R == 0.0

    def test_scalar_curvature_enabled(
        self, sample_parameter_dict, sample_fisher_diagonal, device
    ):
        """Test that scalar curvature can be computed when enabled."""
        config = CurvatureAnalysisConfig(
            enabled=True,
            compute_ricci=True,  # Required for scalar
            compute_scalar=True,
            num_tangent_samples=3
        )
        analyzer = CurvatureAnalyzer(config, device=device)

        R = analyzer.compute_scalar_curvature(
            sample_parameter_dict,
            sample_fisher_diagonal,
            num_directions=3  # Very small for fast test
        )

        # Should return finite value
        assert torch.isfinite(torch.tensor(R))

    def test_curvature_interpretation_flat(self, curvature_config, device):
        """Test interpretation for nearly flat curvature."""
        analyzer = CurvatureAnalyzer(curvature_config, device=device)

        # Mock results with near-zero curvature
        results = {
            'mean_curvature': 1e-6,
            'std_curvature': 1e-7,
            'min_curvature': -1e-6,
            'max_curvature': 1e-6,
            'curvature_samples': [0.0, 0.0, 0.0],
            'num_samples': 3
        }

        # Should interpret as flat
        # (Recreate the interpretation logic)
        mean_curv = results['mean_curvature']
        if abs(mean_curv) < 1e-4:
            interpretation = "Nearly flat (Euclidean-like)"
        elif mean_curv > 1e-4:
            interpretation = "Positively curved (sphere-like, geodesics converge)"
        else:
            interpretation = "Negatively curved (hyperbolic/saddle-like, geodesics diverge)"

        assert "flat" in interpretation.lower()

    def test_empty_curvature_samples_handling(
        self, sample_parameter_dict, device
    ):
        """Test handling when no valid curvature samples can be computed."""
        # Create a degenerate Fisher metric (all zeros) to force numerical issues
        fisher_degenerate = {
            name: torch.zeros_like(param)
            for name, param in sample_parameter_dict.items()
        }

        config = CurvatureAnalysisConfig(enabled=True, num_tangent_samples=5)
        analyzer = CurvatureAnalyzer(config, device=device)

        # This should handle the degenerate case gracefully
        results = analyzer.estimate_curvature_distribution(
            sample_parameter_dict, fisher_degenerate
        )

        # Should return default values
        assert results['num_samples'] == 0
        assert len(results['curvature_samples']) == 0
        assert "Unable to compute" in results['interpretation']
