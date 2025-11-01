"""
Tests for Fisher Information Matrix computation and Riemannian metric operations.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sitv.geometry.config import FisherApproximationType
from sitv.geometry.metric import FisherMetricService


@pytest.fixture
def tiny_model_and_tokenizer():
    """Fixture providing a tiny model for testing.

    Uses a small model to keep tests fast.
    """
    model_name = "gpt2"  # Small model for fast tests
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@pytest.fixture
def sample_texts():
    """Fixture providing sample texts for Fisher computation."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Neural networks process information.",
        "Optimization algorithms converge.",
    ]


class TestFisherMetricService:
    """Test suite for FisherMetricService."""

    def test_identity_metric(self, tiny_model_and_tokenizer, sample_texts):
        """Test that Euclidean metric returns identity (ones)."""
        model, tokenizer = tiny_model_and_tokenizer

        service = FisherMetricService(
            tokenizer=tokenizer,
            device="cpu",
            approximation_type=FisherApproximationType.EUCLIDEAN
        )

        fisher = service.compute_fisher_information_matrix(
            model, sample_texts, batch_size=2
        )

        # Check that all Fisher values are 1 (identity)
        for name, value in fisher.items():
            assert torch.all(value == 1.0), f"Identity metric should be all ones for {name}"

    def test_diagonal_fisher_positive(self, tiny_model_and_tokenizer, sample_texts):
        """Test that diagonal Fisher has positive values."""
        model, tokenizer = tiny_model_and_tokenizer

        service = FisherMetricService(
            tokenizer=tokenizer,
            device="cpu",
            approximation_type=FisherApproximationType.DIAGONAL,
            num_samples=4
        )

        fisher = service.compute_fisher_information_matrix(
            model, sample_texts, batch_size=2
        )

        # Check that all Fisher diagonal elements are positive
        for name, diag in fisher.items():
            assert torch.all(diag > 0), f"Diagonal Fisher should be positive for {name}"

    def test_diagonal_fisher_shape_matches_params(
        self, tiny_model_and_tokenizer, sample_texts
    ):
        """Test that Fisher diagonal has same shape as parameters."""
        model, tokenizer = tiny_model_and_tokenizer

        service = FisherMetricService(
            tokenizer=tokenizer,
            device="cpu",
            approximation_type=FisherApproximationType.DIAGONAL,
            num_samples=4
        )

        fisher = service.compute_fisher_information_matrix(
            model, sample_texts, batch_size=2
        )

        # Check shapes match
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in fisher, f"Fisher missing entry for {name}"
                assert fisher[name].shape == param.shape, (
                    f"Shape mismatch for {name}: "
                    f"Fisher {fisher[name].shape} vs param {param.shape}"
                )

    def test_riemannian_norm_euclidean(self, tiny_model_and_tokenizer):
        """Test Riemannian norm reduces to Euclidean norm for identity metric."""
        model, tokenizer = tiny_model_and_tokenizer

        service = FisherMetricService(
            tokenizer=tokenizer,
            device="cpu",
            approximation_type=FisherApproximationType.EUCLIDEAN
        )

        # Create a simple vector
        vector = {}
        fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                vector[name] = torch.randn_like(param)
                fisher[name] = torch.ones_like(param)

        # Compute norms
        riemannian_norm = service.compute_riemannian_norm(vector, fisher)

        # Compute Euclidean norm manually
        euclidean_norm = torch.sqrt(
            sum(torch.sum(v ** 2).item() for v in vector.values())
        )

        # Should be equal for identity metric
        assert abs(riemannian_norm - euclidean_norm) < 1e-5, (
            "Riemannian norm with identity metric should equal Euclidean norm"
        )

    def test_riemannian_distance_euclidean(self, tiny_model_and_tokenizer):
        """Test Riemannian distance for Euclidean metric."""
        model, tokenizer = tiny_model_and_tokenizer

        service = FisherMetricService(
            tokenizer=tokenizer,
            device="cpu",
            approximation_type=FisherApproximationType.EUCLIDEAN
        )

        # Create two parameter sets
        params1 = {}
        params2 = {}
        fisher = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                params1[name] = torch.randn_like(param)
                params2[name] = torch.randn_like(param)
                fisher[name] = torch.ones_like(param)

        # Compute distance
        distance = service.compute_riemannian_distance(params1, params2, fisher)

        # Should be non-negative
        assert distance >= 0, "Distance should be non-negative"

        # Distance to self should be zero
        distance_to_self = service.compute_riemannian_distance(params1, params1, fisher)
        assert distance_to_self < 1e-5, "Distance to self should be zero"

    def test_fisher_cache(self, tiny_model_and_tokenizer, sample_texts):
        """Test Fisher metric caching."""
        model, tokenizer = tiny_model_and_tokenizer

        service = FisherMetricService(
            tokenizer=tokenizer,
            device="cpu",
            approximation_type=FisherApproximationType.DIAGONAL,
            num_samples=4
        )

        # Initially no cache
        assert service.get_cached_fisher() is None

        # Compute and cache
        fisher = service.compute_fisher_information_matrix(
            model, sample_texts, batch_size=2
        )
        service.cache_fisher(fisher)

        # Retrieve from cache
        cached = service.get_cached_fisher()
        assert cached is not None
        assert len(cached) == len(fisher)

        # Clear cache
        service.clear_cache()
        assert service.get_cached_fisher() is None

    def test_christoffel_symbols_structure(self, tiny_model_and_tokenizer, sample_texts):
        """Test Christoffel symbols computation returns correct structure."""
        model, tokenizer = tiny_model_and_tokenizer

        service = FisherMetricService(
            tokenizer=tokenizer,
            device="cpu",
            approximation_type=FisherApproximationType.DIAGONAL,
            num_samples=4
        )

        fisher = service.compute_fisher_information_matrix(
            model, sample_texts, batch_size=2
        )

        christoffel = service.compute_christoffel_symbols(model, fisher)

        # Check structure
        assert isinstance(christoffel, dict)

        # For constant Fisher, Christoffel should be zero
        for name, value in christoffel.items():
            if not name.startswith("_"):
                assert torch.all(value == 0), (
                    f"Christoffel should be zero for constant Fisher: {name}"
                )

    @pytest.mark.slow
    def test_full_fisher_small_model(self):
        """Test full Fisher computation on a very small model.

        This test is marked as slow because full Fisher is expensive.
        """
        # Create a tiny dummy model
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 2)

            def forward(self, input_ids, labels=None):
                logits = self.linear(input_ids.float())
                if labels is not None:
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, 2), labels.view(-1)
                    )
                    return type('Output', (), {'loss': loss})()
                return type('Output', (), {'logits': logits})()

        model = TinyModel()

        # Dummy tokenizer
        class DummyTokenizer:
            def __call__(self, texts, **kwargs):
                # Return dummy tensors
                batch_size = len(texts)
                return {
                    'input_ids': torch.randint(0, 10, (batch_size, 8)),
                    'attention_mask': torch.ones((batch_size, 8))
                }

        tokenizer = DummyTokenizer()

        service = FisherMetricService(
            tokenizer=tokenizer,
            device="cpu",
            approximation_type=FisherApproximationType.FULL,
            num_samples=2
        )

        texts = ["sample text 1", "sample text 2"]

        # Should not raise for tiny model
        fisher = service.compute_fisher_information_matrix(
            model, texts, batch_size=1
        )

        # Check structure
        assert "_full_matrix" in fisher
        assert "_param_shapes" in fisher
        assert fisher["_full_matrix"].dim() == 2
        # Matrix should be square
        n = fisher["_full_matrix"].size(0)
        assert fisher["_full_matrix"].size(1) == n

    def test_full_fisher_memory_error_large_model(self, tiny_model_and_tokenizer, sample_texts):
        """Test that full Fisher raises MemoryError for large models."""
        model, tokenizer = tiny_model_and_tokenizer

        service = FisherMetricService(
            tokenizer=tokenizer,
            device="cpu",
            approximation_type=FisherApproximationType.FULL
        )

        # GPT2 has >100M parameters, should raise MemoryError
        with pytest.raises(MemoryError, match="Full Fisher matrix would require"):
            service.compute_fisher_information_matrix(
                model, sample_texts, batch_size=2
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
