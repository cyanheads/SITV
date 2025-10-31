"""
Experiment configuration for SITV.

This module provides configuration classes for experiments,
loading parameters from config.yaml as the single source of truth.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


def load_config_yaml(config_path: str | None = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. If None, uses default location.

    Returns:
        Dictionary with configuration values
    """
    if config_path is None:
        # Default: look for config.yaml in project root
        config_file = Path(__file__).parent.parent.parent / "config.yaml"
    else:
        config_file = Path(config_path)

    if not config_file.exists():
        # Return empty dict if no config file exists (use defaults)
        return {}

    with open(config_file) as f:
        return yaml.safe_load(f) or {}


# Load YAML config as single source of truth
_YAML_CONFIG = load_config_yaml()


def reload_config(config_path: str | None = None) -> None:
    """Reload configuration from YAML file.

    Args:
        config_path: Path to config.yaml. If None, uses default location.
    """
    global _YAML_CONFIG
    _YAML_CONFIG = load_config_yaml(config_path)


def _get(keys: str, default):
    """Get nested config value using dot notation (e.g., 'model.name')."""
    value = _YAML_CONFIG
    for key in keys.split('.'):
        if isinstance(value, dict):
            value = value.get(key)
            if value is None:
                return default
        else:
            return default
    return value if value is not None else default


@dataclass
class AlphaSweepConfig:
    """Configuration for alpha sweep experiments.

    Attributes:
        alpha_range: Range of alpha values (min, max)
        num_samples: Number of alpha samples
        enable_squaring_test: Whether to test M(2α)
        threshold: Threshold for zero-crossing detection
    """

    alpha_range: tuple[float, float] = field(
        default_factory=lambda: (
            _get('alpha_sweep.alpha_min', -3.0),
            _get('alpha_sweep.alpha_max', 3.0)
        )
    )
    num_samples: int = field(
        default_factory=lambda: _get('alpha_sweep.num_samples', 150)
    )
    enable_squaring_test: bool = field(
        default_factory=lambda: _get('alpha_sweep.enable_squaring_test', True)
    )
    threshold: float = field(
        default_factory=lambda: _get('alpha_sweep.threshold', 0.1)
    )


@dataclass
class Composition2DConfig:
    """Configuration for 2D composition experiments.

    Attributes:
        alpha_range: Range of alpha values (min, max)
        beta_range: Range of beta values (min, max)
        num_samples_per_dim: Samples per dimension (creates n² grid)
    """

    alpha_range: tuple[float, float] = field(
        default_factory=lambda: (
            _get('composition_2d.alpha_min', -2.0),
            _get('composition_2d.alpha_max', 2.0)
        )
    )
    beta_range: tuple[float, float] = field(
        default_factory=lambda: (
            _get('composition_2d.beta_min', -2.0),
            _get('composition_2d.beta_max', 2.0)
        )
    )
    num_samples_per_dim: int = field(
        default_factory=lambda: _get('composition_2d.num_samples_per_dim', 30)
    )


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning.

    Attributes:
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Training batch size
        max_length: Maximum sequence length
        data_repetition_factor: Multiplier for repeating training examples
        save_strategy: Model save strategy ("no", "steps", "epoch")
        logging_steps: Steps between logging
    """

    num_epochs: int = field(
        default_factory=lambda: _get('fine_tuning.num_epochs', 2)
    )
    learning_rate: float = field(
        default_factory=lambda: _get('fine_tuning.learning_rate', 5e-5)
    )
    batch_size: int = field(
        default_factory=lambda: _get('fine_tuning.batch_size', 16)
    )
    max_length: int = field(
        default_factory=lambda: _get('fine_tuning.max_length', 512)
    )
    data_repetition_factor: int = field(
        default_factory=lambda: _get('fine_tuning.data_repetition_factor', 100)
    )
    save_strategy: str = field(
        default_factory=lambda: _get('fine_tuning.save_strategy', 'no')
    )
    logging_steps: int = field(
        default_factory=lambda: _get('fine_tuning.logging_steps', 10)
    )


@dataclass
class EvaluationConfig:
    """Configuration for evaluation datasets.

    Attributes:
        general_dataset: Name of general evaluation dataset to use
            (e.g., 'mixed_domain', 'wikitext', 'coding', 'common_knowledge')
    """

    general_dataset: str = field(
        default_factory=lambda: _get('evaluation.general_dataset', 'mixed_domain')
    )


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    This is the top-level configuration object that combines
    all experiment parameters.

    Attributes:
        model_name: HuggingFace model identifier
        output_dir: Output directory for results
        device: Device for computation (auto-detected if None)
        task_name: Name of task to run
        analysis_only: Whether to skip fine-tuning and load saved models
        enable_2d_composition: Whether to run 2D composition experiment
        alpha_sweep: Alpha sweep configuration
        composition_2d: 2D composition configuration
        fine_tuning: Fine-tuning configuration
        evaluation: Evaluation configuration
    """

    model_name: str = field(
        default_factory=lambda: _get('model.name', 'google/gemma-3-4b-it')
    )
    output_dir: str = field(
        default_factory=lambda: _get('output.dir', 'outputs')
    )
    device: Optional[str] = field(
        default_factory=lambda: _get('model.device', None)
    )
    task_name: str = field(
        default_factory=lambda: _get('task.name', 'sentiment_positive')
    )
    analysis_only: bool = field(
        default_factory=lambda: _get('output.analysis_only', False)
    )
    enable_2d_composition: bool = field(
        default_factory=lambda: _get('composition_2d.enable', False)
    )

    # Sub-configurations
    alpha_sweep: AlphaSweepConfig = field(default_factory=AlphaSweepConfig)
    composition_2d: Composition2DConfig = field(default_factory=Composition2DConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_args(cls, args) -> "ExperimentConfig":
        """Create configuration from command-line arguments.

        Command-line arguments override config.yaml values. If a custom
        config file path is provided via --config, it will be loaded first.

        Args:
            args: Parsed arguments from argparse

        Returns:
            ExperimentConfig object

        Examples:
            >>> args = parser.parse_args()
            >>> config = ExperimentConfig.from_args(args)
        """
        # Reload config if custom path provided
        if hasattr(args, 'config') and args.config is not None:
            reload_config(args.config)

        # Create config with CLI overrides
        return cls(
            model_name=args.model,
            output_dir=args.output_dir,
            device=args.device,
            task_name=args.task,
            analysis_only=args.analysis_only,
            enable_2d_composition=args.enable_2d,
            alpha_sweep=AlphaSweepConfig(
                alpha_range=(args.alpha_min, args.alpha_max),
                num_samples=args.num_samples,
                enable_squaring_test=args.enable_squaring,
            ),
            composition_2d=Composition2DConfig(
                alpha_range=(args.alpha_2d_min, args.alpha_2d_max),
                beta_range=(args.beta_min, args.beta_max),
                num_samples_per_dim=args.samples_2d,
            ),
            fine_tuning=FineTuningConfig(
                num_epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                data_repetition_factor=args.data_repetition_factor,
            ),
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration

        Examples:
            >>> config = ExperimentConfig()
            >>> config_dict = config.to_dict()
        """
        return {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "device": self.device,
            "task_name": self.task_name,
            "analysis_only": self.analysis_only,
            "enable_2d_composition": self.enable_2d_composition,
            "alpha_sweep": {
                "alpha_range": self.alpha_sweep.alpha_range,
                "num_samples": self.alpha_sweep.num_samples,
                "enable_squaring_test": self.alpha_sweep.enable_squaring_test,
                "threshold": self.alpha_sweep.threshold,
            },
            "composition_2d": {
                "alpha_range": self.composition_2d.alpha_range,
                "beta_range": self.composition_2d.beta_range,
                "num_samples_per_dim": self.composition_2d.num_samples_per_dim,
            },
            "fine_tuning": {
                "num_epochs": self.fine_tuning.num_epochs,
                "learning_rate": self.fine_tuning.learning_rate,
                "batch_size": self.fine_tuning.batch_size,
                "max_length": self.fine_tuning.max_length,
                "save_strategy": self.fine_tuning.save_strategy,
                "logging_steps": self.fine_tuning.logging_steps,
            },
        }
