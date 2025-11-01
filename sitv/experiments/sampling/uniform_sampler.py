"""
Uniform sampler for alpha sweep experiments.

This module implements the uniform (linear) sampling strategy,
which is the default behavior of the original alpha sweep.
"""

import numpy as np
from typing import List, Optional
from sitv.data.models import AlphaSweepResult
from sitv.experiments.sampling.base_sampler import BaseSampler


class UniformSampler(BaseSampler):
    """Uniform sampling strategy using linear spacing.

    This sampler generates evenly spaced alpha values across the
    specified range using np.linspace(). This is the default sampling
    strategy and matches the original alpha sweep behavior.

    Example:
        >>> sampler = UniformSampler(alpha_range=(-1.0, 2.0), num_samples=100)
        >>> alphas = sampler.generate_samples()
        >>> print(len(alphas))  # 100
        >>> print(alphas[0], alphas[-1])  # -1.0, 2.0
    """

    def generate_samples(self, results: Optional[List[AlphaSweepResult]] = None) -> np.ndarray:
        """Generate uniformly spaced alpha values.

        Args:
            results: Previously collected results (ignored for uniform sampling)

        Returns:
            Array of evenly spaced alpha values
        """
        return np.linspace(self.alpha_min, self.alpha_max, self.num_samples)

    def should_continue(self, results: List[AlphaSweepResult]) -> bool:
        """Check if more samples are needed.

        For uniform sampling, we generate all samples at once,
        so this always returns False.

        Args:
            results: Results collected so far

        Returns:
            False (all samples generated upfront)
        """
        return False
