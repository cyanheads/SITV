"""Tests for experiment configuration."""
import pytest
import tempfile
from pathlib import Path
from sitv.experiments.config import (
    ExperimentConfig,
    AlphaSweepConfig,
    FineTuningConfig,
    SamplingConfig,
    GradientAnalysisConfig,
    Composition2DConfig,
    EvaluationConfig,
    load_config_yaml,
    reload_config,
)


class TestConfigYAMLLoading:
    """Tests for YAML configuration loading."""

    def test_load_nonexistent_config_returns_empty_dict(self):
        """Should return empty dict when config file doesn't exist."""
        result = load_config_yaml("/nonexistent/path/config.yaml")
        assert result == {}

    def test_load_valid_config(self, tmp_path):
        """Should load valid YAML configuration."""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
model:
  name: "test-model"
  device: "cpu"

task:
  name: "test_task"

alpha_sweep:
  alpha_min: -2.0
  alpha_max: 2.0
  num_samples: 50
  enable_squaring_test: true
""")

        config = load_config_yaml(str(config_file))
        assert config["model"]["name"] == "test-model"
        assert config["model"]["device"] == "cpu"
        assert config["task"]["name"] == "test_task"
        assert config["alpha_sweep"]["num_samples"] == 50

    def test_reload_config(self, tmp_path):
        """Should reload configuration from new file."""
        config_file = tmp_path / "reload_test.yaml"
        config_file.write_text("""
model:
  name: "reloaded-model"
""")

        reload_config(str(config_file))
        config = ExperimentConfig()
        assert config.model_name == "reloaded-model"


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_default_values(self):
        """Should create config with default values."""
        config = ExperimentConfig()

        # Check defaults are set
        assert config.model_name == "google/gemma-3-4b-it"
        assert config.output_dir == "outputs"
        assert config.task_name == "sentiment_positive"
        assert config.analysis_only is False
        assert config.enable_2d_composition is False

    def test_to_dict(self):
        """Should convert config to dictionary."""
        config = ExperimentConfig()
        config_dict = config.to_dict()

        assert "model_name" in config_dict
        assert "alpha_sweep" in config_dict
        assert "fine_tuning" in config_dict
        assert config_dict["model_name"] == "google/gemma-3-4b-it"

    def test_from_args_with_analysis_only(self):
        """Should set analysis_only from args."""
        # Mock args object
        class Args:
            config = None
            analysis_only = True

        config = ExperimentConfig.from_args(Args())
        assert config.analysis_only is True

    def test_from_args_without_analysis_only(self):
        """Should keep analysis_only as False if not in args."""
        class Args:
            config = None
            analysis_only = False

        config = ExperimentConfig.from_args(Args())
        assert config.analysis_only is False

    def test_from_args_with_custom_config(self, tmp_path):
        """Should load custom config file from args."""
        config_file = tmp_path / "custom.yaml"
        config_file.write_text("""
model:
  name: "custom-model"
task:
  name: "custom_task"
""")

        class Args:
            config = str(config_file)
            analysis_only = False

        config = ExperimentConfig.from_args(Args())
        assert config.model_name == "custom-model"
        assert config.task_name == "custom_task"


class TestAlphaSweepConfig:
    """Tests for AlphaSweepConfig."""

    def test_default_values(self):
        """Should create alpha sweep config with defaults."""
        config = AlphaSweepConfig()

        assert config.alpha_range == (-3.0, 3.0)
        assert config.num_samples == 150
        assert config.enable_squaring_test is True
        assert config.threshold == 0.1

    def test_includes_sampling_config(self):
        """Should include sampling configuration."""
        config = AlphaSweepConfig()

        assert hasattr(config, 'sampling')
        assert isinstance(config.sampling, SamplingConfig)
        assert config.sampling.strategy == "uniform"

    def test_includes_gradient_analysis_config(self):
        """Should include gradient analysis configuration."""
        config = AlphaSweepConfig()

        assert hasattr(config, 'gradient_analysis')
        assert isinstance(config.gradient_analysis, GradientAnalysisConfig)


class TestFineTuningConfig:
    """Tests for FineTuningConfig."""

    def test_default_values(self):
        """Should create fine-tuning config with defaults."""
        config = FineTuningConfig()

        assert config.num_epochs == 2
        assert config.learning_rate == 5e-5
        assert config.batch_size == 16
        assert config.max_length == 512
        assert config.data_repetition_factor == 100
        assert config.save_strategy == "no"
        assert config.logging_steps == 10


class TestSamplingConfig:
    """Tests for SamplingConfig."""

    def test_default_values(self):
        """Should create sampling config with defaults."""
        config = SamplingConfig()

        assert config.strategy == "uniform"
        assert config.adaptive_coarse_samples == 20
        assert config.adaptive_refine_factor == 3
        assert config.adaptive_curvature_threshold == 0.5
        assert config.bayesian_n_initial == 10
        assert config.bayesian_acquisition == "ei"


class TestGradientAnalysisConfig:
    """Tests for GradientAnalysisConfig."""

    def test_default_values(self):
        """Should create gradient analysis config with defaults."""
        config = GradientAnalysisConfig()

        assert config.enable is False
        assert config.smooth_sigma == 0.5
        assert config.gradient_threshold == 0.01
        assert config.curvature_threshold == 0.001


class TestComposition2DConfig:
    """Tests for Composition2DConfig."""

    def test_default_values(self):
        """Should create 2D composition config with defaults."""
        config = Composition2DConfig()

        assert config.alpha_range == (-2.0, 2.0)
        assert config.beta_range == (-2.0, 2.0)
        assert config.num_samples_per_dim == 30


class TestEvaluationConfig:
    """Tests for EvaluationConfig."""

    def test_default_values(self):
        """Should create evaluation config with defaults."""
        config = EvaluationConfig()

        assert config.general_dataset == "mixed_domain"


class TestConfigIntegration:
    """Integration tests for configuration loading."""

    def test_full_config_from_yaml(self, tmp_path):
        """Should load complete configuration from YAML."""
        config_file = tmp_path / "full_config.yaml"
        config_file.write_text("""
model:
  name: "gpt2"
  device: "cuda"

task:
  name: "sentiment_negative"

output:
  dir: "custom_outputs"
  analysis_only: false

alpha_sweep:
  alpha_min: -1.0
  alpha_max: 1.0
  num_samples: 25
  enable_squaring_test: false
  threshold: 0.05
  sampling_strategy: "adaptive"
  adaptive_coarse_samples: 10
  adaptive_refine_factor: 2

fine_tuning:
  num_epochs: 1
  learning_rate: 1e-4
  batch_size: 8
  max_length: 256
  data_repetition_factor: 50

evaluation:
  general_dataset: "wikitext"

composition_2d:
  enable: true
  alpha_min: -1.0
  alpha_max: 1.0
  beta_min: -1.0
  beta_max: 1.0
  num_samples_per_dim: 20
""")

        reload_config(str(config_file))
        config = ExperimentConfig()

        # Verify all values are loaded correctly
        assert config.model_name == "gpt2"
        assert config.device == "cuda"
        assert config.task_name == "sentiment_negative"
        assert config.output_dir == "custom_outputs"
        assert config.alpha_sweep.alpha_range == (-1.0, 1.0)
        assert config.alpha_sweep.num_samples == 25
        assert config.alpha_sweep.enable_squaring_test is False
        assert config.alpha_sweep.sampling.strategy == "adaptive"
        assert config.fine_tuning.num_epochs == 1
        assert config.fine_tuning.batch_size == 8
        assert config.evaluation.general_dataset == "wikitext"
        assert config.enable_2d_composition is True
        assert config.composition_2d.num_samples_per_dim == 20
