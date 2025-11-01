"""
Tests for geodesic integration on parameter manifolds.
"""

import pytest
import torch

from sitv.geometry.config import GeodesicIntegrationConfig
from sitv.geometry.geodesic import GeodesicIntegrator


@pytest.fixture
def simple_config():
    """Fixture providing basic geodesic integration configuration."""
    return GeodesicIntegrationConfig(
        enabled=True,
        num_steps=100,
        tolerance=1e-6,
        step_size_control=False,
        max_iterations=1000
    )


@pytest.fixture
def sample_parameter_dict():
    """Fixture providing sample parameter dictionaries."""
    base_point = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 10),
    }

    tangent_vector = {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 10),
    }

    return base_point, tangent_vector


class TestGeodesicIntegrator:
    """Test suite for GeodesicIntegrator."""

    def test_euclidean_exponential_map_at_zero(
        self, simple_config, sample_parameter_dict
    ):
        """Test that exp_p(0·v) = p (identity)."""
        integrator = GeodesicIntegrator(simple_config, device="cpu")
        base_point, tangent_vector = sample_parameter_dict

        result = integrator.exponential_map(
            base_point, tangent_vector, t=0.0
        )

        # At t=0, should return base point
        for name in base_point:
            assert torch.allclose(result[name], base_point[name], atol=1e-6), (
                f"exp_p(0·v) should equal p for {name}"
            )

    def test_euclidean_exponential_map_linearity(
        self, simple_config, sample_parameter_dict
    ):
        """Test that Euclidean exp_p(t·v) = p + t·v."""
        integrator = GeodesicIntegrator(simple_config, device="cpu")
        base_point, tangent_vector = sample_parameter_dict

        t = 0.5

        result = integrator.exponential_map(
            base_point, tangent_vector, t=t
        )

        # For Euclidean (no Fisher metric), should be straight line
        for name in base_point:
            expected = base_point[name] + t * tangent_vector[name]
            assert torch.allclose(result[name], expected, atol=1e-6), (
                f"Euclidean exponential should be p + t·v for {name}"
            )

    def test_exponential_map_shape_preservation(
        self, simple_config, sample_parameter_dict
    ):
        """Test that exponential map preserves parameter shapes."""
        integrator = GeodesicIntegrator(simple_config, device="cpu")
        base_point, tangent_vector = sample_parameter_dict

        result = integrator.exponential_map(
            base_point, tangent_vector, t=1.0
        )

        # Shapes should match
        for name in base_point:
            assert result[name].shape == base_point[name].shape, (
                f"Shape mismatch for {name}: "
                f"{result[name].shape} vs {base_point[name].shape}"
            )

    def test_exponential_map_scaling(
        self, simple_config, sample_parameter_dict
    ):
        """Test that exp_p(2t·v) = exp_p(t·(2v)) for Euclidean case."""
        integrator = GeodesicIntegrator(simple_config, device="cpu")
        base_point, tangent_vector = sample_parameter_dict

        t = 0.5

        # exp_p(2t·v)
        result1 = integrator.exponential_map(
            base_point, tangent_vector, t=2*t
        )

        # exp_p(t·(2v))
        scaled_vector = {
            name: 2 * vec for name, vec in tangent_vector.items()
        }
        result2 = integrator.exponential_map(
            base_point, scaled_vector, t=t
        )

        # Should be equal for Euclidean space
        for name in base_point:
            assert torch.allclose(result1[name], result2[name], atol=1e-6), (
                f"Scaling property violated for {name}"
            )

    def test_log_map_inverse_of_exp(
        self, simple_config, sample_parameter_dict
    ):
        """Test that log_p(exp_p(v)) ≈ v."""
        integrator = GeodesicIntegrator(simple_config, device="cpu")
        base_point, tangent_vector = sample_parameter_dict

        # Compute exp_p(v)
        target = integrator.exponential_map(
            base_point, tangent_vector, t=1.0
        )

        # Compute log_p(target)
        recovered_vector = integrator.log_map(base_point, target)

        # Should recover original tangent vector
        for name in tangent_vector:
            assert torch.allclose(
                recovered_vector[name], tangent_vector[name], atol=1e-5
            ), f"Log map should be inverse of exp map for {name}"

    def test_log_map_at_base_point(
        self, simple_config, sample_parameter_dict
    ):
        """Test that log_p(p) = 0."""
        integrator = GeodesicIntegrator(simple_config, device="cpu")
        base_point, _ = sample_parameter_dict

        # log_p(p) should be zero vector
        log_vector = integrator.log_map(base_point, base_point)

        for name in base_point:
            assert torch.allclose(
                log_vector[name], torch.zeros_like(base_point[name]), atol=1e-6
            ), f"log_p(p) should be zero for {name}"

    def test_parallel_transport_identity_euclidean(
        self, simple_config, sample_parameter_dict
    ):
        """Test that parallel transport is identity in Euclidean space."""
        integrator = GeodesicIntegrator(simple_config, device="cpu")
        base_point, tangent_vector = sample_parameter_dict

        # Create a vector to transport
        vector_to_transport = {
            name: torch.randn_like(param)
            for name, param in base_point.items()
        }

        # Transport along tangent direction
        transported = integrator.parallel_transport(
            vector_to_transport,
            base_point,
            tangent_vector,
            t=0.5,
            christoffel=None  # Euclidean
        )

        # In Euclidean space, parallel transport is identity
        for name in vector_to_transport:
            assert torch.allclose(
                transported[name], vector_to_transport[name], atol=1e-6
            ), f"Parallel transport should be identity in Euclidean space for {name}"

    def test_fixed_step_rk_consistency(
        self, simple_config, sample_parameter_dict
    ):
        """Test that RK4 integration is consistent across multiple calls."""
        integrator = GeodesicIntegrator(simple_config, device="cpu")
        base_point, tangent_vector = sample_parameter_dict

        # Compute exponential map twice with same inputs
        result1 = integrator.exponential_map(
            base_point, tangent_vector, t=1.0
        )

        result2 = integrator.exponential_map(
            base_point, tangent_vector, t=1.0
        )

        # Should be identical (deterministic)
        for name in base_point:
            assert torch.allclose(result1[name], result2[name], atol=1e-10), (
                f"RK4 integration should be deterministic for {name}"
            )

    def test_exponential_map_with_zero_vector(
        self, simple_config, sample_parameter_dict
    ):
        """Test exp_p(t·0) = p for any t."""
        integrator = GeodesicIntegrator(simple_config, device="cpu")
        base_point, _ = sample_parameter_dict

        zero_vector = {
            name: torch.zeros_like(param)
            for name, param in base_point.items()
        }

        for t in [0.0, 0.5, 1.0, 2.0]:
            result = integrator.exponential_map(
                base_point, zero_vector, t=t
            )

            # Should remain at base point
            for name in base_point:
                assert torch.allclose(result[name], base_point[name], atol=1e-6), (
                    f"exp_p(t·0) should equal p for t={t}, {name}"
                )

    def test_exponential_map_negative_t(
        self, simple_config, sample_parameter_dict
    ):
        """Test that exp_p(-t·v) goes in opposite direction."""
        integrator = GeodesicIntegrator(simple_config, device="cpu")
        base_point, tangent_vector = sample_parameter_dict

        t = 0.5

        # exp_p(t·v)
        result_forward = integrator.exponential_map(
            base_point, tangent_vector, t=t
        )

        # exp_p(-t·v)
        result_backward = integrator.exponential_map(
            base_point, tangent_vector, t=-t
        )

        # Distance from base should be equal
        for name in base_point:
            dist_forward = torch.norm(result_forward[name] - base_point[name])
            dist_backward = torch.norm(result_backward[name] - base_point[name])
            assert torch.allclose(dist_forward, dist_backward, atol=1e-5), (
                f"Forward and backward distances should be equal for {name}"
            )

    @pytest.mark.parametrize("num_steps", [10, 50, 100, 200])
    def test_exponential_map_convergence_with_steps(
        self, sample_parameter_dict, num_steps
    ):
        """Test that exponential map converges as num_steps increases.

        For Euclidean case, result should be exact regardless of steps.
        """
        config = GeodesicIntegrationConfig(
            enabled=True,
            num_steps=num_steps,
            tolerance=1e-6,
            step_size_control=False
        )

        integrator = GeodesicIntegrator(config, device="cpu")
        base_point, tangent_vector = sample_parameter_dict

        result = integrator.exponential_map(
            base_point, tangent_vector, t=1.0
        )

        # For Euclidean, should match exact solution p + t·v
        for name in base_point:
            expected = base_point[name] + tangent_vector[name]
            # Allow small numerical error for integration
            assert torch.allclose(result[name], expected, atol=1e-4), (
                f"Convergence test failed for {num_steps} steps on {name}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
