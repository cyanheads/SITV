"""Tests for task vector computation."""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock
from sitv.core.task_vector import TaskVectorService


class SimpleModel(nn.Module):
    """Simple test model."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestComputeTaskVector:
    """Tests for compute_task_vector function."""

    def test_basic_computation(self):
        """Should compute task vector as finetuned - base."""
        # Create two identical models
        base_model = SimpleModel()
        finetuned_model = SimpleModel()

        # Make sure they start with same weights
        finetuned_model.load_state_dict(base_model.state_dict())

        # Modify finetuned model weights slightly
        with torch.no_grad():
            for param in finetuned_model.parameters():
                param.add_(0.1)  # Add 0.1 to all parameters

        # Compute task vector
        task_vector = TaskVectorService.compute(base_model, finetuned_model)

        # Task vector should be approximately 0.1 for all parameters
        for name, tv_param in task_vector.items():
            assert torch.allclose(tv_param, torch.full_like(tv_param, 0.1), atol=1e-6)

    def test_zero_task_vector(self):
        """Should produce zero task vector for identical models."""
        base_model = SimpleModel()
        finetuned_model = SimpleModel()

        # Load same weights
        finetuned_model.load_state_dict(base_model.state_dict())

        task_vector = TaskVectorService.compute(base_model, finetuned_model)

        # Task vector should be all zeros
        for name, tv_param in task_vector.items():
            assert torch.allclose(tv_param, torch.zeros_like(tv_param), atol=1e-6)

    def test_handles_different_devices_cpu_to_cpu(self):
        """Should handle models on CPU correctly."""
        base_model = SimpleModel().to("cpu")
        finetuned_model = SimpleModel().to("cpu")

        # Should not raise
        task_vector = TaskVectorService.compute(base_model, finetuned_model)
        assert len(task_vector) > 0

    def test_meta_device_raises_error(self):
        """Should raise error if model has meta device parameters."""
        base_model = SimpleModel()
        finetuned_model = SimpleModel()

        # Mock a parameter to be on meta device
        mock_param = Mock()
        mock_param.device.type = 'meta'
        mock_param.detach.return_value.cpu.return_value = torch.randn(10, 5)

        # Override named_parameters to return mock
        finetuned_model.named_parameters = lambda: [
            ("test_param", mock_param)
        ]
        base_model.named_parameters = lambda: [
            ("test_param", torch.randn(10, 5))
        ]

        with pytest.raises(RuntimeError, match="meta device"):
            TaskVectorService.compute(base_model, finetuned_model)

    def test_output_on_cpu(self):
        """Task vector should always be on CPU."""
        base_model = SimpleModel()
        finetuned_model = SimpleModel()

        task_vector = TaskVectorService.compute(base_model, finetuned_model)

        for name, tv_param in task_vector.items():
            assert tv_param.device.type == "cpu"
