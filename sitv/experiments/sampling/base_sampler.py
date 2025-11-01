"""
Base sampler interface for alpha sweep experiments.

This module defines the abstract base class for all sampling strategies.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from sitv.data.models import AlphaSweepResult


class BaseSampler(ABC):
    """Abstract base class for alpha sampling strategies.

    All samplers should implement this interface to generate alpha values
    for evaluation in alpha sweep experiments.

    Attributes:
        alpha_range: Tuple of (min, max) alpha values
        num_samples: Target number of samples to generate
    """

    def __init__(self, alpha_range: Tuple[float, float], num_samples: int):
        """Initialize the sampler.

        Args:
            alpha_range: Range of alpha values (min, max)
            num_samples: Target number of samples
        """
        self.alpha_range = alpha_range
        self.num_samples = num_samples
        self.alpha_min, self.alpha_max = alpha_range

    @abstractmethod
    def generate_samples(self, results: Optional[List[AlphaSweepResult]] = None) -> np.ndarray:
        """Generate alpha values to evaluate.

        Args:
            results: Previously collected results (for adaptive sampling)

        Returns:
            Array of alpha values to evaluate

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement generate_samples()")

    @abstractmethod
    def should_continue(self, results: List[AlphaSweepResult]) -> bool:
        """Decide if sampling should continue.

        Args:
            results: Results collected so far

        Returns:
            True if more samples should be collected, False otherwise

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement should_continue()")

    def get_name(self) -> str:
        """Get sampler name for logging.

        Returns:
            Human-readable sampler name
        """
        return self.__class__.__name__

    def get_config(self) -> dict:
        """Get sampler configuration for metadata.

        Returns:
            Dictionary with sampler configuration
        """
        return {
            "sampler": self.get_name(),
            "alpha_range": self.alpha_range,
            "num_samples": self.num_samples,
        }
