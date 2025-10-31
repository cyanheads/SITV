"""
Sampling strategies for alpha sweep experiments.

This module provides different sampling strategies for efficiently
exploring the loss landscape along task vector directions.
"""

from sitv.experiments.sampling.base_sampler import BaseSampler
from sitv.experiments.sampling.uniform_sampler import UniformSampler
from sitv.experiments.sampling.adaptive_sampler import AdaptiveSampler
from sitv.experiments.sampling.bayesian_sampler import BayesianSampler

__all__ = [
    "BaseSampler",
    "UniformSampler",
    "AdaptiveSampler",
    "BayesianSampler",
]
