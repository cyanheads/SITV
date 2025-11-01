"""
Progress tracking utilities for SITV experiments.

This module provides utilities for tracking and displaying progress
during long-running operations.
"""

import time
from datetime import datetime
from typing import List, Optional
from transformers import TrainerCallback
from sitv.utils.timing import format_eta, calculate_avg_time


class ProgressTracker:
    """Track progress of iterative operations with ETA calculation.

    This tracker handles progress reporting with automatic ETA calculation
    based on historical timing data.

    Attributes:
        total: Total number of iterations
        current: Current iteration
        times: List of elapsed times per iteration
        start_time: Overall start timestamp

    Examples:
        >>> tracker = ProgressTracker(total=100)
        >>> for i in range(100):
        ...     tracker.start_iteration()
        ...     # ... do work ...
        ...     elapsed = tracker.end_iteration()
        ...     print(tracker.get_status())
    """

    def __init__(self, total: int):
        """Initialize the progress tracker.

        Args:
            total: Total number of iterations
        """
        self.total = total
        self.current = 0
        self.times: List[float] = []
        self.start_time: Optional[float] = None
        self.iteration_start: Optional[float] = None

    def start_iteration(self) -> None:
        """Mark the start of an iteration."""
        if self.start_time is None:
            self.start_time = time.time()
        self.iteration_start = time.time()

    def end_iteration(self) -> float:
        """Mark the end of an iteration.

        Returns:
            Time elapsed for this iteration
        """
        if self.iteration_start is None:
            return 0.0

        elapsed = time.time() - self.iteration_start
        self.times.append(elapsed)
        self.current += 1
        return elapsed

    def get_progress_pct(self) -> float:
        """Get progress as percentage.

        Returns:
            Progress percentage (0-100)
        """
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100

    def get_eta_string(self) -> str:
        """Get formatted ETA string.

        Returns:
            ETA string or "calculating..." if not enough data
        """
        if not self.times:
            return "ETA: calculating..."

        avg_time = calculate_avg_time(self.times)
        remaining = self.total - self.current
        return format_eta(avg_time, remaining)

    def get_status(self) -> str:
        """Get complete status string.

        Returns:
            Formatted status string with progress and ETA

        Examples:
            >>> tracker.get_status()
            '[25/100] (25.0%) | ETA: 1.5m'
        """
        pct = self.get_progress_pct()
        eta = self.get_eta_string()
        return f"[{self.current}/{self.total}] ({pct:.1f}%) | {eta}"

    def get_total_elapsed(self) -> float:
        """Get total elapsed time since start.

        Returns:
            Total elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def is_complete(self) -> bool:
        """Check if tracking is complete.

        Returns:
            True if current >= total
        """
        return self.current >= self.total


class FineTuningProgressCallback(TrainerCallback):
    """Custom callback for detailed fine-tuning progress reporting.

    This callback provides detailed progress tracking during model fine-tuning,
    including ETA calculation, loss tracking, and training history.

    Attributes:
        training_history: List of training log entries
        epoch_start_time: Timestamp when current epoch started
        step_times: List of timestamps for each step

    Examples:
        >>> from transformers import Trainer
        >>> callback = FineTuningProgressCallback()
        >>> trainer = Trainer(model=model, callbacks=[callback])
        >>> trainer.train()
    """

    def __init__(self):
        """Initialize the fine-tuning progress callback."""
        self.training_history = []
        self.epoch_start_time = None
        self.step_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch.

        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            **kwargs: Additional keyword arguments
        """
        self.epoch_start_time = time.time()
        epoch_num = int(state.epoch) if state.epoch else 0
        print(f"\n{'─' * 60}")
        print(f"Epoch {epoch_num + 1}/{args.num_train_epochs}")
        print(f"{'─' * 60}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs information.

        Args:
            args: TrainingArguments
            state: TrainerState
            control: TrainerControl
            logs: Dictionary of logged values
            **kwargs: Additional keyword arguments
        """
        if logs:
            step_time = time.time()
            self.step_times.append(step_time)

            # Calculate ETA
            if len(self.step_times) > 1 and state.max_steps:
                avg_step_time = (self.step_times[-1] - self.step_times[0]) / len(self.step_times)
                remaining_steps = state.max_steps - state.global_step
                eta_seconds = avg_step_time * remaining_steps
                eta_str = f"{eta_seconds / 60:.1f}m" if eta_seconds > 60 else f"{eta_seconds:.0f}s"
            else:
                eta_str = "calculating..."

            # Record history
            log_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                "timestamp": datetime.now().isoformat(),
                **logs,
            }
            self.training_history.append(log_entry)

            # Print progress
            print(f"  Step {state.global_step}/{state.max_steps} | ", end="")
            if "loss" in logs:
                print(f"Loss: {logs['loss']:.4f} | ", end="")
            if "learning_rate" in logs:
                print(f"LR: {logs['learning_rate']:.2e} | ", end="")
            if "grad_norm" in logs:
                print(f"Grad: {logs['grad_norm']:.2f} | ", end="")
            print(f"ETA: {eta_str}")
