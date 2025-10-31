"""
Pytest configuration and shared fixtures for SITV tests.
"""
import pytest
import torch
from dataclasses import dataclass
from typing import Dict, Any


@pytest.fixture
def device():
    """Provide a device for testing (CPU to avoid GPU requirements)."""
    return torch.device("cpu")


@pytest.fixture
def mock_model_state_dict():
    """Provide a mock model state dict for testing."""
    return {
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
        "layer2.weight": torch.randn(5, 10),
        "layer2.bias": torch.randn(5),
    }


@pytest.fixture
def mock_task_definition():
    """Provide a mock task definition for testing."""
    from sitv.data.models import TaskDefinition

    return TaskDefinition(
        name="test_task",
        train_texts=["Example text 1", "Example text 2"],
        eval_texts=["Test evaluation text 1", "Test evaluation text 2"],
        description="Test task for unit testing",
    )


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary output directory for testing."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def mock_alpha_sweep_result():
    """Provide a mock AlphaSweepResult for testing."""
    from sitv.data.models import AlphaSweepResult
    import numpy as np

    alphas = np.linspace(-2, 2, 10)
    # Return a list of AlphaSweepResult objects instead of a single object with lists
    # to match the new data model structure
    results = []
    for alpha in alphas:
        results.append(
            AlphaSweepResult(
                alpha=float(alpha),
                loss=0.5 + 0.1 * abs(alpha),
                base_loss=0.6,
                functional_return=abs(0.5 + 0.1 * abs(alpha) - 0.6),
                task_eval_loss=0.5 + 0.1 * abs(alpha),
            )
        )
    return results
