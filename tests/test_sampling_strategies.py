"""Tests for sampling strategies."""
import pytest
import numpy as np
from sitv.data.models import AlphaSweepResult
from sitv.experiments.sampling.base_sampler import BaseSampler
from sitv.experiments.sampling.uniform_sampler import UniformSampler
from sitv.experiments.sampling.adaptive_sampler import AdaptiveSampler


class TestBaseSampler:
    """Tests for BaseSampler abstract class."""

    def test_cannot_instantiate_directly(self):
        """Should not be able to instantiate abstract class."""
        with pytest.raises(TypeError):
            BaseSampler(alpha_range=(-1.0, 1.0), num_samples=10)

    def test_get_name(self):
        """Should return class name."""
        sampler = UniformSampler(alpha_range=(-1.0, 1.0), num_samples=10)
        assert sampler.get_name() == "UniformSampler"

    def test_get_config(self):
        """Should return configuration dictionary."""
        sampler = UniformSampler(alpha_range=(-2.0, 3.0), num_samples=50)
        config = sampler.get_config()

        assert "sampler" in config
        assert "alpha_range" in config
        assert "num_samples" in config
        assert config["alpha_range"] == (-2.0, 3.0)
        assert config["num_samples"] == 50


class TestUniformSampler:
    """Tests for UniformSampler."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        sampler = UniformSampler(alpha_range=(-1.0, 2.0), num_samples=100)

        assert sampler.alpha_range == (-1.0, 2.0)
        assert sampler.num_samples == 100
        assert sampler.alpha_min == -1.0
        assert sampler.alpha_max == 2.0

    def test_generate_samples_correct_count(self):
        """Should generate correct number of samples."""
        sampler = UniformSampler(alpha_range=(-1.0, 1.0), num_samples=50)
        samples = sampler.generate_samples()

        assert len(samples) == 50

    def test_generate_samples_correct_range(self):
        """Should generate samples in correct range."""
        sampler = UniformSampler(alpha_range=(-2.0, 3.0), num_samples=100)
        samples = sampler.generate_samples()

        assert samples[0] == pytest.approx(-2.0)
        assert samples[-1] == pytest.approx(3.0)
        assert np.all(samples >= -2.0)
        assert np.all(samples <= 3.0)

    def test_generate_samples_evenly_spaced(self):
        """Should generate evenly spaced samples."""
        sampler = UniformSampler(alpha_range=(0.0, 10.0), num_samples=11)
        samples = sampler.generate_samples()

        expected_spacing = 1.0
        spacings = np.diff(samples)

        assert np.allclose(spacings, expected_spacing)

    def test_generate_samples_ignores_results(self):
        """Should ignore previous results (not adaptive)."""
        sampler = UniformSampler(alpha_range=(-1.0, 1.0), num_samples=10)

        # Generate with no results
        samples1 = sampler.generate_samples()

        # Generate with mock results
        mock_results = [
            AlphaSweepResult(
                alpha=0.0, loss=1.0, base_loss=1.0,
                functional_return=0.0, task_performance=1.0
            )
        ]
        samples2 = sampler.generate_samples(results=mock_results)

        # Should be identical
        assert np.allclose(samples1, samples2)

    def test_should_continue_always_false(self):
        """Should always return False (all samples generated upfront)."""
        sampler = UniformSampler(alpha_range=(-1.0, 1.0), num_samples=10)

        assert sampler.should_continue([]) is False

        mock_results = [
            AlphaSweepResult(
                alpha=0.0, loss=1.0, base_loss=1.0,
                functional_return=0.0, task_performance=1.0
            )
        ]
        assert sampler.should_continue(mock_results) is False


class TestAdaptiveSampler:
    """Tests for AdaptiveSampler."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        sampler = AdaptiveSampler(
            alpha_range=(-1.0, 2.0),
            num_samples=100,
            coarse_samples=20,
            refine_factor=3,
            curvature_threshold=0.5
        )

        assert sampler.alpha_range == (-1.0, 2.0)
        assert sampler.num_samples == 100
        assert sampler.coarse_samples == 20
        assert sampler.refine_factor == 3
        assert sampler.curvature_threshold == 0.5

    def test_coarse_pass(self):
        """Should generate coarse samples first."""
        sampler = AdaptiveSampler(
            alpha_range=(-1.0, 1.0),
            num_samples=100,
            coarse_samples=20
        )

        samples = sampler.generate_samples()

        assert len(samples) == 20
        assert samples[0] == pytest.approx(-1.0)
        assert samples[-1] == pytest.approx(1.0)

    def test_should_continue_after_coarse(self):
        """Should indicate refinement needed after coarse pass."""
        sampler = AdaptiveSampler(
            alpha_range=(-1.0, 1.0),
            num_samples=100,
            coarse_samples=10
        )

        # Generate coarse samples
        sampler.generate_samples()

        # Should want to continue for refinement
        mock_results = [
            AlphaSweepResult(
                alpha=float(i)/10, loss=1.0, base_loss=1.0,
                functional_return=0.0, task_performance=1.0
            )
            for i in range(10)
        ]
        assert sampler.should_continue(mock_results) is True

    def test_refinement_pass(self):
        """Should generate refinement samples in second pass."""
        sampler = AdaptiveSampler(
            alpha_range=(-1.0, 1.0),
            num_samples=100,
            coarse_samples=10,
            refine_factor=2
        )

        # First pass: coarse
        coarse_samples = sampler.generate_samples()
        assert len(coarse_samples) == 10

        # Create mock results with interesting region
        results = []
        for alpha in coarse_samples:
            # Create a "valley" at alpha=0
            loss = 2.0 + (alpha ** 2)
            results.append(AlphaSweepResult(
                alpha=float(alpha),
                loss=loss,
                base_loss=2.0,
                functional_return=abs(loss - 2.0),
                task_performance=loss
            ))

        # Second pass: refinement
        refinement_samples = sampler.generate_samples(results=results)

        # Should generate some refinement samples
        assert len(refinement_samples) > 0

    def test_find_high_curvature_regions(self):
        """Should identify high curvature regions."""
        sampler = AdaptiveSampler(
            alpha_range=(-2.0, 2.0),
            num_samples=100,
            curvature_threshold=0.1
        )

        # Create data with high curvature at center
        alphas = np.linspace(-2.0, 2.0, 20)
        losses = np.array([alpha ** 2 for alpha in alphas])  # Parabola

        high_curvature = sampler._find_high_curvature_regions(alphas, losses)

        # Should find regions in the high-curvature area
        assert len(high_curvature) > 0

    def test_find_zero_crossing_regions(self):
        """Should identify zero-crossing regions."""
        sampler = AdaptiveSampler(
            alpha_range=(-2.0, 2.0),
            num_samples=100
        )

        # Create data that crosses base_loss
        alphas = np.linspace(-2.0, 2.0, 20)
        base_loss = 1.0
        losses = np.array([base_loss + alpha for alpha in alphas])  # Linear

        crossings = sampler._find_zero_crossing_regions(alphas, losses, base_loss)

        # Should find crossing near alpha=0
        assert len(crossings) > 0

    def test_find_extrema_regions(self):
        """Should identify local minima and maxima."""
        sampler = AdaptiveSampler(
            alpha_range=(-2.0, 2.0),
            num_samples=100
        )

        # Create data with local minimum at 0
        alphas = np.linspace(-2.0, 2.0, 20)
        losses = np.array([alpha ** 2 for alpha in alphas])  # Parabola, min at 0

        extrema = sampler._find_extrema_regions(alphas, losses)

        # Should find the minimum
        assert len(extrema) > 0

    def test_merge_regions(self):
        """Should merge nearby regions."""
        sampler = AdaptiveSampler(
            alpha_range=(-2.0, 2.0),
            num_samples=100,
            refinement_window=0.5
        )

        # Nearby regions that should merge
        regions = [0.0, 0.1, 0.2, 1.5, 1.6]
        merged = sampler._merge_regions(regions)

        # Should have fewer regions after merging
        assert len(merged) < len(regions)

    def test_refine_region(self):
        """Should generate refinement samples around region."""
        sampler = AdaptiveSampler(
            alpha_range=(-2.0, 2.0),
            num_samples=100,
            refine_factor=2,
            refinement_window=0.3
        )

        center = 0.0
        existing_alphas = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        new_samples = sampler._refine_region(center, existing_alphas)

        # Should generate some new samples
        assert len(new_samples) > 0

        # New samples should be near center
        for sample in new_samples:
            assert abs(sample - center) < 1.0

    def test_get_config_includes_adaptive_params(self):
        """Should include adaptive-specific parameters in config."""
        sampler = AdaptiveSampler(
            alpha_range=(-1.0, 1.0),
            num_samples=100,
            coarse_samples=15,
            refine_factor=4,
            curvature_threshold=0.7
        )

        config = sampler.get_config()

        assert config["coarse_samples"] == 15
        assert config["refine_factor"] == 4
        assert config["curvature_threshold"] == 0.7
        assert "refinement_window" in config

    def test_empty_refinement(self):
        """Should handle case where no interesting regions are found."""
        sampler = AdaptiveSampler(
            alpha_range=(-1.0, 1.0),
            num_samples=100,
            coarse_samples=10,
            curvature_threshold=99.0  # Very high threshold
        )

        # First pass
        coarse_samples = sampler.generate_samples()

        # Create flat results (no interesting regions)
        results = []
        for alpha in coarse_samples:
            results.append(AlphaSweepResult(
                alpha=float(alpha),
                loss=2.0,  # Constant loss
                base_loss=2.0,
                functional_return=0.0,
                task_performance=2.0
            ))

        # Second pass should return empty
        refinement_samples = sampler.generate_samples(results=results)
        assert len(refinement_samples) == 0


class TestSamplingStrategiesIntegration:
    """Integration tests comparing sampling strategies."""

    def test_uniform_vs_adaptive_coverage(self):
        """Should verify both strategies cover the alpha range."""
        alpha_range = (-2.0, 2.0)
        num_samples = 50

        # Uniform sampler
        uniform = UniformSampler(alpha_range=alpha_range, num_samples=num_samples)
        uniform_samples = uniform.generate_samples()

        # Adaptive sampler - just coarse pass
        adaptive = AdaptiveSampler(
            alpha_range=alpha_range,
            num_samples=num_samples,
            coarse_samples=20
        )
        adaptive_samples = adaptive.generate_samples()

        # Both should cover the range
        assert uniform_samples.min() == pytest.approx(alpha_range[0])
        assert uniform_samples.max() == pytest.approx(alpha_range[1])
        assert adaptive_samples.min() == pytest.approx(alpha_range[0])
        assert adaptive_samples.max() == pytest.approx(alpha_range[1])

    def test_adaptive_reduces_total_samples(self):
        """Should verify adaptive uses fewer samples for same coverage."""
        alpha_range = (-2.0, 2.0)

        uniform = UniformSampler(alpha_range=alpha_range, num_samples=100)
        uniform_samples = uniform.generate_samples()

        adaptive = AdaptiveSampler(
            alpha_range=alpha_range,
            num_samples=100,
            coarse_samples=30
        )
        adaptive_coarse = adaptive.generate_samples()

        # Adaptive coarse pass should use significantly fewer samples
        assert len(adaptive_coarse) < len(uniform_samples)
        assert len(adaptive_coarse) == 30
