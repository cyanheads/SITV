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
    from main import TaskDefinition

    return TaskDefinition(
        name="test_task",
        train_texts=["Example text 1", "Example text 2"],
        train_labels=[1, 0],
        eval_text="Test evaluation text",
        correct_label=1,
        wrong_label=0,
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
    from main import AlphaSweepResult
    import numpy as np

    alphas = np.linspace(-2, 2, 10)
    return AlphaSweepResult(
        alphas=alphas.tolist(),
        losses=[0.5 + 0.1 * abs(a) for a in alphas],
        correct_probs=[0.7 - 0.05 * abs(a) for a in alphas],
        wrong_probs=[0.3 + 0.05 * abs(a) for a in alphas],
    )
