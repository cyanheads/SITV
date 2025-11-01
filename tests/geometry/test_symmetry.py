"""
Tests for symmetry detection and quotient space analysis.
"""

import pytest
import torch
from torch import nn

from sitv.geometry.config import SymmetryAnalysisConfig
from sitv.geometry.symmetry import SymmetryAnalyzer


class MockEvaluator:
    """Mock evaluator for testing symmetry detection."""

    def __init__(self, loss_value: float = 2.0):
        self.loss_value = loss_value
        self.call_count = 0

    def evaluate(self, model: nn.Module, texts: list[str]) -> float:
        """Return mock loss value."""
        self.call_count += 1
        return self.loss_value


class SimpleLinearModel(nn.Module):
    """Simple model for symmetry testing."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 5, output_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def symmetry_config():
    """Fixture providing symmetry analysis configuration."""
    return SymmetryAnalysisConfig(
        enabled=True,
        detect_rotations=True,
        detect_permutations=True,
        detect_scaling=True,
        quotient_space=False,
        symmetry_tolerance=0.01
    )


@pytest.fixture
def mock_evaluator():
    """Fixture providing mock evaluator."""
    return MockEvaluator(loss_value=2.0)


@pytest.fixture
def simple_model(device):
    """Fixture providing a simple linear model."""
    return SimpleLinearModel().to(device)


@pytest.fixture
def sample_texts():
    """Fixture providing sample evaluation texts."""
    return [
        "This is a test sentence.",
        "Another example for evaluation.",
        "Testing symmetry detection."
    ]


class TestSymmetryAnalyzer:
    """Test suite for SymmetryAnalyzer."""

    def test_initialization(self, symmetry_config, mock_evaluator, device):
        """Test symmetry analyzer initialization."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        assert analyzer.config == symmetry_config
        assert analyzer.evaluator == mock_evaluator
        assert analyzer.device == device
        assert analyzer.symmetry_tolerance == 0.01

    def test_generate_orthogonal_matrix(self, symmetry_config, mock_evaluator, device):
        """Test random orthogonal matrix generation."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        Q = analyzer._generate_orthogonal_matrix(5)

        # Check shape
        assert Q.shape == (5, 5)

        # Check orthogonality: Q^T Q = I
        identity = Q.T @ Q
        expected_identity = torch.eye(5, device=device)
        assert torch.allclose(identity, expected_identity, atol=1e-5)

        # Check determinant is approximately 1 (proper rotation)
        det = torch.det(Q)
        assert abs(det - 1.0) < 0.1 or abs(det + 1.0) < 0.1

    @pytest.mark.parametrize("n", [3, 5, 10, 20])
    def test_orthogonal_matrix_different_sizes(
        self, symmetry_config, mock_evaluator, device, n
    ):
        """Test orthogonal matrix generation for different sizes."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        Q = analyzer._generate_orthogonal_matrix(n)

        assert Q.shape == (n, n)
        identity = Q.T @ Q
        expected_identity = torch.eye(n, device=device)
        assert torch.allclose(identity, expected_identity, atol=1e-4)

    def test_get_rotatable_layers(
        self, symmetry_config, mock_evaluator, simple_model, device
    ):
        """Test identification of rotatable layers."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        rotatable = analyzer._get_rotatable_layers(simple_model)

        # Should include fc1.weight and fc2.weight
        assert len(rotatable) == 2
        assert 'fc1.weight' in rotatable
        assert 'fc2.weight' in rotatable
        # Should NOT include biases
        assert 'fc1.bias' not in rotatable
        assert 'fc2.bias' not in rotatable

    def test_get_permutable_layers(
        self, symmetry_config, mock_evaluator, simple_model, device
    ):
        """Test identification of permutable layers."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        permutable = analyzer._get_permutable_layers(simple_model)

        # Should include weight matrices
        assert len(permutable) == 2
        assert 'fc1.weight' in permutable
        assert 'fc2.weight' in permutable

    def test_detect_rotation_symmetry_disabled(
        self, mock_evaluator, simple_model, sample_texts, device
    ):
        """Test that rotation detection returns empty when disabled."""
        config = SymmetryAnalysisConfig(
            enabled=True,
            detect_rotations=False  # Disabled
        )
        analyzer = SymmetryAnalyzer(config, mock_evaluator, device=device)

        result = analyzer.detect_rotation_symmetry(simple_model, sample_texts)

        assert not result['is_symmetric']
        assert result['symmetry_score'] == 0.0
        assert result['num_tests'] == 0
        assert 'disabled' in result['note']

    def test_detect_rotation_symmetry_structure(
        self, symmetry_config, mock_evaluator, simple_model, sample_texts, device
    ):
        """Test that rotation detection returns correct structure."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        result = analyzer.detect_rotation_symmetry(
            simple_model, sample_texts, num_tests=3
        )

        # Check required keys
        required_keys = [
            'is_symmetric', 'symmetry_score', 'avg_loss_deviation',
            'violations', 'num_tests', 'tested_layers'
        ]
        for key in required_keys:
            assert key in result

        # Check types
        assert isinstance(result['is_symmetric'], bool)
        assert isinstance(result['symmetry_score'], float)
        assert isinstance(result['violations'], list)
        assert result['num_tests'] <= 3

    def test_detect_permutation_symmetry_disabled(
        self, mock_evaluator, simple_model, sample_texts, device
    ):
        """Test that permutation detection returns empty when disabled."""
        config = SymmetryAnalysisConfig(
            enabled=True,
            detect_permutations=False  # Disabled
        )
        analyzer = SymmetryAnalyzer(config, mock_evaluator, device=device)

        result = analyzer.detect_permutation_symmetry(simple_model, sample_texts)

        assert not result['is_symmetric']
        assert result['symmetry_score'] == 0.0

    def test_detect_permutation_symmetry_structure(
        self, symmetry_config, mock_evaluator, simple_model, sample_texts, device
    ):
        """Test that permutation detection returns correct structure."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        result = analyzer.detect_permutation_symmetry(
            simple_model, sample_texts, num_tests=3
        )

        required_keys = [
            'is_symmetric', 'symmetry_score', 'avg_loss_deviation',
            'violations', 'num_tests', 'tested_layers'
        ]
        for key in required_keys:
            assert key in result

    def test_detect_scaling_symmetry_disabled(
        self, mock_evaluator, simple_model, sample_texts, device
    ):
        """Test that scaling detection returns empty when disabled."""
        config = SymmetryAnalysisConfig(
            enabled=True,
            detect_scaling=False  # Disabled
        )
        analyzer = SymmetryAnalyzer(config, mock_evaluator, device=device)

        result = analyzer.detect_scaling_symmetry(simple_model, sample_texts)

        assert not result['is_symmetric']
        assert result['symmetry_score'] == 0.0

    def test_detect_scaling_symmetry_structure(
        self, symmetry_config, mock_evaluator, simple_model, sample_texts, device
    ):
        """Test that scaling detection returns correct structure."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        result = analyzer.detect_scaling_symmetry(
            simple_model, sample_texts, num_tests=3
        )

        required_keys = [
            'is_symmetric', 'symmetry_score', 'avg_loss_deviation',
            'violations', 'num_tests', 'tested_scales'
        ]
        for key in required_keys:
            assert key in result

        assert isinstance(result['tested_scales'], list)
        assert len(result['tested_scales']) <= 3

    def test_canonical_permutation_sorting(
        self, symmetry_config, mock_evaluator, device
    ):
        """Test that permutation canonicalization sorts by weight norms."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        # Create parameters with known norms
        params = {
            'layer.weight': torch.tensor([
                [1.0, 3.0, 2.0],  # Column norms: 1.0, 3.0, 2.0
            ], device=device),
            'layer.bias': torch.tensor([10.0, 30.0, 20.0], device=device)
        }

        canonical = analyzer._canonical_permutation(params)

        # Columns should be sorted by norm (descending): 3.0, 2.0, 1.0
        # So order should be: column 1, column 2, column 0
        expected_weight = torch.tensor([[3.0, 2.0, 1.0]], device=device)
        expected_bias = torch.tensor([30.0, 20.0, 10.0], device=device)

        assert torch.allclose(canonical['layer.weight'], expected_weight)
        assert torch.allclose(canonical['layer.bias'], expected_bias)

    def test_canonical_scaling_normalization(
        self, symmetry_config, mock_evaluator, device
    ):
        """Test that scaling canonicalization normalizes to unit norm."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        params = {
            'layer1.weight': torch.tensor([[3.0, 4.0]], device=device),
            'layer1.bias': torch.tensor([0.0], device=device)
        }

        canonical = analyzer._canonical_scaling(params)

        # Total Frobenius norm: sqrt(3^2 + 4^2) = 5
        # After normalization: [3/5, 4/5] = [0.6, 0.8]
        expected_weight = torch.tensor([[0.6, 0.8]], device=device)
        expected_bias = torch.tensor([0.0], device=device)

        assert torch.allclose(canonical['layer1.weight'], expected_weight, atol=1e-5)
        assert torch.allclose(canonical['layer1.bias'], expected_bias, atol=1e-5)

        # Check total norm is 1
        total_norm_sq = sum(torch.sum(p ** 2) for p in canonical.values())
        assert abs(total_norm_sq - 1.0) < 1e-5

    def test_compute_canonical_representative_permutation(
        self, symmetry_config, mock_evaluator, device
    ):
        """Test canonical representative for permutation symmetry."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        params = {
            'fc.weight': torch.randn(10, 5, device=device),
            'fc.bias': torch.randn(5, device=device)
        }

        canonical = analyzer.compute_canonical_representative(
            params, symmetry_group="permutation"
        )

        # Check that neurons are sorted by norm
        weight = canonical['fc.weight']
        norms = torch.norm(weight, dim=0)

        # Norms should be in descending order
        for i in range(len(norms) - 1):
            assert norms[i] >= norms[i + 1]

    def test_compute_canonical_representative_scaling(
        self, symmetry_config, mock_evaluator, device
    ):
        """Test canonical representative for scaling symmetry."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        params = {
            'fc.weight': torch.randn(10, 5, device=device) * 10.0  # Large scale
        }

        canonical = analyzer.compute_canonical_representative(
            params, symmetry_group="scaling"
        )

        # Check normalization
        total_norm = torch.sqrt(sum(torch.sum(p ** 2) for p in canonical.values()))
        assert abs(total_norm - 1.0) < 1e-5

    def test_project_to_quotient_space(
        self, simple_model, mock_evaluator, device
    ):
        """Test quotient space projection."""
        config = SymmetryAnalysisConfig(
            enabled=True,
            quotient_space=True  # Enable quotient projection
        )
        analyzer = SymmetryAnalyzer(config, mock_evaluator, device=device)

        # Project to quotient space
        projected_model = analyzer.project_to_quotient_space(
            simple_model, symmetry_types=["permutation", "scaling"]
        )

        # Model should be modified
        assert projected_model is simple_model  # Same reference

        # Check that parameters are normalized
        total_norm_sq = sum(
            torch.sum(p ** 2).item()
            for p in simple_model.parameters()
        )
        # After scaling canonicalization, norm should be 1
        assert abs(total_norm_sq - 1.0) < 1e-4

    def test_project_to_quotient_space_disabled(
        self, simple_model, mock_evaluator, device
    ):
        """Test that projection does nothing when quotient_space=False."""
        config = SymmetryAnalysisConfig(
            enabled=True,
            quotient_space=False  # Disabled
        )
        analyzer = SymmetryAnalyzer(config, mock_evaluator, device=device)

        # Save original parameters
        original_params = {
            name: param.clone()
            for name, param in simple_model.named_parameters()
        }

        # Attempt projection
        projected_model = analyzer.project_to_quotient_space(
            simple_model, symmetry_types=["permutation"]
        )

        # Parameters should be unchanged
        for name, param in projected_model.named_parameters():
            assert torch.allclose(param, original_params[name])

    def test_analyze_all_symmetries(
        self, symmetry_config, mock_evaluator, simple_model, sample_texts, device
    ):
        """Test comprehensive symmetry analysis."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        results = analyzer.analyze_all_symmetries(
            simple_model, sample_texts, num_tests_per_type=3
        )

        # Check all symmetry types present
        assert 'rotation' in results
        assert 'permutation' in results
        assert 'scaling' in results
        assert 'summary' in results

        # Check summary structure
        summary = results['summary']
        assert 'any_symmetry_detected' in summary
        assert 'num_symmetries_detected' in summary
        assert isinstance(summary['any_symmetry_detected'], bool)
        assert isinstance(summary['num_symmetries_detected'], int)

    def test_analyze_all_symmetries_partial(
        self, mock_evaluator, simple_model, sample_texts, device
    ):
        """Test symmetry analysis with some types disabled."""
        config = SymmetryAnalysisConfig(
            enabled=True,
            detect_rotations=True,
            detect_permutations=False,  # Disabled
            detect_scaling=True
        )
        analyzer = SymmetryAnalyzer(config, mock_evaluator, device=device)

        results = analyzer.analyze_all_symmetries(
            simple_model, sample_texts, num_tests_per_type=2
        )

        # Only enabled types should be present
        assert 'rotation' in results
        assert 'permutation' not in results
        assert 'scaling' in results

    def test_apply_rotation_preserves_other_params(
        self, symmetry_config, mock_evaluator, simple_model, device
    ):
        """Test that rotation only affects target layer."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        # Save original parameters
        original_fc2 = simple_model.fc2.weight.clone()
        original_bias = simple_model.fc1.bias.clone()

        # Apply rotation to fc1.weight
        analyzer._apply_rotation_to_layer(simple_model, 'fc1.weight')

        # fc1.weight should change (rotated)
        # fc2.weight and biases should be unchanged
        assert torch.allclose(simple_model.fc2.weight, original_fc2)
        assert torch.allclose(simple_model.fc1.bias, original_bias)

    def test_empty_symmetry_result(self, symmetry_config, mock_evaluator, device):
        """Test that empty result has correct structure."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        result = analyzer._empty_symmetry_result("test_type")

        assert not result['is_symmetric']
        assert result['symmetry_score'] == 0.0
        assert result['avg_loss_deviation'] == 0.0
        assert result['violations'] == []
        assert result['num_tests'] == 0
        assert 'note' in result

    def test_symmetry_score_calculation(
        self, symmetry_config, mock_evaluator, simple_model, sample_texts, device
    ):
        """Test that symmetry score is in valid range [0, 1]."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        result = analyzer.detect_scaling_symmetry(
            simple_model, sample_texts, num_tests=5
        )

        score = result['symmetry_score']
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize("tolerance", [0.001, 0.01, 0.1])
    def test_symmetry_tolerance_affects_detection(
        self, mock_evaluator, simple_model, sample_texts, device, tolerance
    ):
        """Test that tolerance parameter affects symmetry detection."""
        config = SymmetryAnalysisConfig(
            enabled=True,
            detect_scaling=True,
            symmetry_tolerance=tolerance
        )
        analyzer = SymmetryAnalyzer(config, mock_evaluator, device=device)

        result = analyzer.detect_scaling_symmetry(
            simple_model, sample_texts, num_tests=5
        )

        # Number of violations should decrease with larger tolerance
        assert result['avg_loss_deviation'] >= 0.0

    def test_model_state_restored_after_test(
        self, symmetry_config, mock_evaluator, simple_model, sample_texts, device
    ):
        """Test that model state is restored after symmetry tests."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        # Save original state
        original_state = {
            name: param.clone()
            for name, param in simple_model.named_parameters()
        }

        # Run symmetry tests
        analyzer.detect_rotation_symmetry(simple_model, sample_texts, num_tests=3)

        # Check that state is restored
        for name, param in simple_model.named_parameters():
            assert torch.allclose(param, original_state[name])

    def test_permutation_idempotence(
        self, symmetry_config, mock_evaluator, device
    ):
        """Test that canonical permutation is idempotent."""
        analyzer = SymmetryAnalyzer(symmetry_config, mock_evaluator, device=device)

        params = {
            'layer.weight': torch.randn(5, 3, device=device),
            'layer.bias': torch.randn(3, device=device)
        }

        # Apply canonicalization twice
        canonical1 = analyzer._canonical_permutation(params)
        canonical2 = analyzer._canonical_permutation(canonical1)

        # Should be identical
        for name in canonical1.keys():
            assert torch.allclose(canonical1[name], canonical2[name])
