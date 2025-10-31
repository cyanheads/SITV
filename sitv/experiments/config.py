"""
Experiment configuration for SITV.

This module provides configuration classes for experiments,
allowing externalization of parameters and easy modification.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AlphaSweepConfig:
    """Configuration for alpha sweep experiments.

    Attributes:
        alpha_range: Range of alpha values (min, max)
        num_samples: Number of alpha samples
        enable_squaring_test: Whether to test M(2α)
        threshold: Threshold for zero-crossing detection
    """

    alpha_range: tuple[float, float] = (-3.0, 3.0)
    num_samples: int = 100
    enable_squaring_test: bool = True
    threshold: float = 0.1


@dataclass
class Composition2DConfig:
    """Configuration for 2D composition experiments.

    Attributes:
        alpha_range: Range of alpha values (min, max)
        beta_range: Range of beta values (min, max)
        num_samples_per_dim: Samples per dimension (creates n² grid)
    """

    alpha_range: tuple[float, float] = (-2.0, 2.0)
    beta_range: tuple[float, float] = (-2.0, 2.0)
    num_samples_per_dim: int = 20


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning.

    Attributes:
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Training batch size
        max_length: Maximum sequence length
        save_strategy: Model save strategy ("no", "steps", "epoch")
        logging_steps: Steps between logging
    """

    num_epochs: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 4
    max_length: int = 128
    save_strategy: str = "no"
    logging_steps: int = 10


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
    """

    model_name: str = "Qwen/Qwen2.5-0.5B"
    output_dir: str = "outputs"
    device: Optional[str] = None  # Auto-detect if None
    task_name: str = "sentiment_positive"
    analysis_only: bool = False
    enable_2d_composition: bool = False

    # Sub-configurations
    alpha_sweep: AlphaSweepConfig = field(default_factory=AlphaSweepConfig)
    composition_2d: Composition2DConfig = field(default_factory=Composition2DConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)

    @classmethod
    def from_args(cls, args) -> "ExperimentConfig":
        """Create configuration from command-line arguments.

        Args:
            args: Parsed arguments from argparse

        Returns:
            ExperimentConfig object

        Examples:
            >>> args = parser.parse_args()
            >>> config = ExperimentConfig.from_args(args)
        """
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
