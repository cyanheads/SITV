"""
Timing utilities for SITV experiments.

This module provides utilities for timing operations and calculating ETAs.
"""

import time
from typing import List, Optional


class Timer:
    """Simple timer for tracking operation duration.

    Attributes:
        start_time: Start timestamp
        end_time: End timestamp

    Examples:
        >>> timer = Timer()
        >>> timer.start()
        >>> # ... do work ...
        >>> elapsed = timer.stop()
        >>> print(f"Took {elapsed:.2f}s")
    """

    def __init__(self):
        """Initialize the timer."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def start(self) -> float:
        """Start the timer.

        Returns:
            Start timestamp
        """
        self.start_time = time.time()
        return self.start_time

    def stop(self) -> float:
        """Stop the timer.

        Returns:
            Elapsed time in seconds
        """
        self.end_time = time.time()
        if self.start_time is None:
            return 0.0
        return self.end_time - self.start_time

    def elapsed(self) -> float:
        """Get elapsed time (even if timer not stopped).

        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = None
        self.end_time = None


def format_eta(avg_time: float, remaining: int) -> str:
    """Format ETA string from average time and remaining iterations.

    Args:
        avg_time: Average time per iteration
        remaining: Remaining iterations

    Returns:
        Formatted ETA string

    Examples:
        >>> format_eta(2.5, 30)
        'ETA: 1.2m'
        >>> format_eta(0.5, 10)
        'ETA: 5.0s'
    """
    eta_seconds = avg_time * remaining
    if eta_seconds > 60:
        return f"ETA: {eta_seconds / 60:.1f}m"
    else:
        return f"ETA: {eta_seconds:.0f}s"


def calculate_avg_time(times: List[float]) -> float:
    """Calculate average time from a list of times.

    Args:
        times: List of time measurements

    Returns:
        Average time

    Examples:
        >>> calculate_avg_time([1.0, 2.0, 3.0])
        2.0
    """
    if not times:
        return 0.0
    return sum(times) / len(times)
