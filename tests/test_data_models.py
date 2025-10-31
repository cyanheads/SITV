"""Tests for data models and dataclasses."""
import pytest
from dataclasses import asdict
import json
from main import (
    AlphaSweepResult,
    TaskDefinition,
    TwoDSweepResult,
    ExperimentMetrics,
)


class TestAlphaSweepResult:
    """Tests for AlphaSweepResult dataclass."""

    def test_creation_with_defaults(self):
        """Should create with default values for optional fields."""
        result = AlphaSweepResult(
            alpha=1.0,
            loss=0.5,
            base_loss=0.6,
            functional_return=0.1,
            task_performance=0.55,
        )
        assert result.alpha == 1.0
        assert result.loss == 0.5
        assert result.loss_2alpha == 0.0  # default
        assert result.perplexity == 0.0  # default

    def test_serialization_to_dict(self):
        """Should serialize to dict correctly."""
        result = AlphaSweepResult(
            alpha=1.0,
            loss=0.5,
            base_loss=0.6,
            functional_return=0.1,
            task_performance=0.55,
            loss_2alpha=0.7,
            functional_return_2alpha=0.15,
            perplexity=1.65,
            perplexity_2alpha=2.01,
        )
        result_dict = asdict(result)
        assert result_dict["alpha"] == 1.0
        assert result_dict["loss_2alpha"] == 0.7
        assert result_dict["perplexity"] == 1.65

    def test_json_serialization(self):
        """Should serialize to JSON successfully."""
        result = AlphaSweepResult(
            alpha=1.0,
            loss=0.5,
            base_loss=0.6,
            functional_return=0.1,
            task_performance=0.55,
        )
        json_str = json.dumps(asdict(result))
        assert "alpha" in json_str
        assert "1.0" in json_str


class TestTaskDefinition:
    """Tests for TaskDefinition dataclass."""

    def test_creation_minimal(self):
        """Should create with minimal required fields."""
        task = TaskDefinition(
            name="test_task",
            train_texts=["text1", "text2"],
            eval_texts=["eval1"],
        )
        assert task.name == "test_task"
        assert len(task.train_texts) == 2
        assert task.description == ""  # default

    def test_creation_with_description(self):
        """Should create with optional description."""
        task = TaskDefinition(
            name="sentiment",
            train_texts=["positive example"],
            eval_texts=["test example"],
            description="Sentiment classification task",
        )
        assert task.description == "Sentiment classification task"


class TestTwoDSweepResult:
    """Tests for TwoDSweepResult dataclass."""

    def test_creation(self):
        """Should create 2D sweep result correctly."""
        result = TwoDSweepResult(
            alpha=1.0,
            beta=0.5,
            loss=0.6,
            base_loss=0.65,
            functional_return=0.05,
            perplexity=1.82,
        )
        assert result.alpha == 1.0
        assert result.beta == 0.5
        assert result.perplexity == 1.82


class TestExperimentMetrics:
    """Tests for ExperimentMetrics dataclass."""

    def test_creation_with_defaults(self):
        """Should create with many default values."""
        metrics = ExperimentMetrics(start_time="2025-01-01T00:00:00")
        assert metrics.start_time == "2025-01-01T00:00:00"
        assert metrics.duration_seconds == 0.0
        assert metrics.model_name == ""
        assert metrics.training_history == []
        assert metrics.zero_crossing_alphas == []

    def test_lists_are_mutable(self):
        """Should allow appending to list fields."""
        metrics = ExperimentMetrics(start_time="2025-01-01T00:00:00")
        metrics.training_history.append({"step": 1, "loss": 0.5})
        metrics.zero_crossing_alphas.append(1.5)
        assert len(metrics.training_history) == 1
        assert len(metrics.zero_crossing_alphas) == 1

    def test_serialization(self):
        """Should serialize complex metrics to dict."""
        metrics = ExperimentMetrics(
            start_time="2025-01-01T00:00:00",
            model_name="test-model",
            num_alpha_samples=100,
            alpha_range=(-3.0, 3.0),
        )
        metrics_dict = asdict(metrics)
        assert metrics_dict["model_name"] == "test-model"
        assert metrics_dict["alpha_range"] == (-3.0, 3.0)
