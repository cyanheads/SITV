"""Experiment orchestration for SITV."""

from sitv.experiments.base import Experiment
from sitv.experiments.alpha_sweep import AlphaSweepExperiment
from sitv.experiments.composition_2d import Composition2DExperiment
from sitv.experiments.orchestrator import ExperimentOrchestrator
from sitv.experiments.config import (
    ExperimentConfig,
    AlphaSweepConfig,
    SamplingConfig,
    GradientAnalysisConfig,
    Composition2DConfig,
    FineTuningConfig,
    EvaluationConfig,
)
from sitv.experiments.sampling import (
    BaseSampler,
    UniformSampler,
    AdaptiveSampler,
    BayesianSampler,
)

__all__ = [
    "Experiment",
    "AlphaSweepExperiment",
    "Composition2DExperiment",
    "ExperimentOrchestrator",
    # Configuration
    "ExperimentConfig",
    "AlphaSweepConfig",
    "SamplingConfig",
    "GradientAnalysisConfig",
    "Composition2DConfig",
    "FineTuningConfig",
    "EvaluationConfig",
    # Sampling strategies
    "BaseSampler",
    "UniformSampler",
    "AdaptiveSampler",
    "BayesianSampler",
]
