"""
Data models and structures for SITV experiments.

This module contains all dataclasses and callbacks used throughout the SITV project.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from transformers import TrainerCallback


@dataclass
class AlphaSweepResult:
    """Store results for a specific α value along the line M(α) = M_base + αT.

    This directly answers: "What does the loss look like as we move along
    the task vector direction?"

    Attributes:
        alpha: Scaling factor α
        loss: L(M_base + αT)
        base_loss: L(M_base) - reference point
        functional_return: |L(M_base + αT) - L(M_base)|
        task_performance: Task-specific loss at M_base + αT
        loss_2alpha: L(M_base + 2αT) - test [W(λ)]² = I analog
        functional_return_2alpha: |L(M_base + 2αT) - L(M_base)|
        perplexity: exp(loss)
        perplexity_2alpha: exp(loss_2alpha)
    """

    alpha: float
    loss: float
    base_loss: float
    functional_return: float
    task_performance: float

    # Squaring test results
    loss_2alpha: float = 0.0
    functional_return_2alpha: float = 0.0

    # Additional metrics
    perplexity: float = 0.0
    perplexity_2alpha: float = 0.0

    # Per-category losses (optional, populated when using "combined" dataset)
    category_losses: dict[str, float] = field(default_factory=dict)


@dataclass
class TaskDefinition:
    """Definition for a task in multi-task experiments.

    Attributes:
        name: Task name (e.g., "sentiment", "summarization")
        train_texts: Training examples
        eval_texts: Task-specific evaluation texts
        description: Human-readable description
    """

    name: str
    train_texts: list[str]
    eval_texts: list[str]
    description: str = ""


@dataclass
class TwoDSweepResult:
    """Store results for 2D composition M(α,β) = M_base + α·T1 + β·T2.

    Attributes:
        alpha: Scaling factor for T1
        beta: Scaling factor for T2
        loss: L(M_base + α·T1 + β·T2)
        base_loss: L(M_base) - reference point
        functional_return: |L - L_base|
        perplexity: exp(loss)
    """

    alpha: float
    beta: float
    loss: float
    base_loss: float
    functional_return: float
    perplexity: float = 0.0


@dataclass
class ExperimentMetrics:
    """Comprehensive metrics for the entire experiment.

    This dataclass tracks all timing, performance, and result metrics
    from a complete SITV experiment run.

    Attributes:
        start_time: ISO format timestamp of experiment start
        end_time: ISO format timestamp of experiment end
        duration_seconds: Total experiment duration
        model_name: Name of the base model used
        device: Device used for computation (cuda/mps/cpu)
        model_parameters: Total number of parameters in model
        finetuning_start_time: When fine-tuning began
        finetuning_end_time: When fine-tuning completed
        finetuning_duration_seconds: Fine-tuning duration
        training_examples: Number of training examples
        num_epochs: Number of training epochs
        learning_rate: Learning rate used
        final_training_loss: Final loss after training
        training_steps: Total training steps
        training_history: List of training log entries
        task_vector_magnitude: L2 norm of task vector
        task_vector_computation_time: Time to compute task vector
        sweep_start_time: When alpha sweep began
        sweep_end_time: When alpha sweep completed
        sweep_duration_seconds: Alpha sweep duration
        num_alpha_samples: Number of alpha values sampled
        alpha_range: Range of alpha values (min, max)
        time_per_alpha_seconds: Average time per alpha sample
        min_general_loss_alpha: Alpha value with minimum general loss
        min_general_loss: Minimum general loss value
        min_task_loss_alpha: Alpha value with minimum task loss
        min_task_loss: Minimum task loss value
        num_zero_crossings: Number of zero-crossings detected
        zero_crossing_alphas: List of alpha values at zero-crossings
        enable_squaring_test: Whether squaring test was enabled
        num_squaring_return_points: Number of points where L(2α) ≈ L(0)
        squaring_return_alphas: List of alphas with squaring return
        task_name: Name of task run
        multi_task_mode: Whether multi-task comparison was run
        enable_2d_composition: Whether 2D composition was enabled
        task_vector_2_magnitude: L2 norm of second task vector (for 2D)
    """

    start_time: str
    end_time: str = ""
    duration_seconds: float = 0.0

    # Model info
    model_name: str = ""
    device: str = ""
    model_parameters: int = 0

    # Fine-tuning metrics
    finetuning_start_time: str = ""
    finetuning_end_time: str = ""
    finetuning_duration_seconds: float = 0.0
    training_examples: int = 0
    num_epochs: int = 0
    learning_rate: float = 0.0
    final_training_loss: float = 0.0
    training_steps: int = 0
    training_history: list[dict] = field(default_factory=list)

    # Task vector metrics
    task_vector_magnitude: float = 0.0
    task_vector_computation_time: float = 0.0

    # Alpha sweep metrics
    sweep_start_time: str = ""
    sweep_end_time: str = ""
    sweep_duration_seconds: float = 0.0
    num_alpha_samples: int = 0
    alpha_range: tuple[float, float] = (-3.0, 3.0)
    time_per_alpha_seconds: float = 0.0

    # Results summary
    min_general_loss_alpha: float = 0.0
    min_general_loss: float = 0.0
    min_task_loss_alpha: float = 0.0
    min_task_loss: float = 0.0
    num_zero_crossings: int = 0
    zero_crossing_alphas: list[float] = field(default_factory=list)

    # Squaring test metrics
    enable_squaring_test: bool = False
    num_squaring_return_points: int = 0
    squaring_return_alphas: list[float] = field(default_factory=list)

    # Multi-task metrics
    task_name: str = "sentiment"
    general_eval_dataset: str = "mixed_domain"  # Which general eval dataset was used
    multi_task_mode: bool = False

    # 2D composition metrics
    enable_2d_composition: bool = False
    task_vector_2_magnitude: float = 0.0


class FineTuningProgressCallback(TrainerCallback):
    """Custom callback for detailed fine-tuning progress reporting.

    This callback provides real-time progress updates during model fine-tuning,
    including ETA calculations, loss tracking, and epoch timing.

    Attributes:
        training_history: List of training log entries
        epoch_start_time: Timestamp of current epoch start
        step_times: List of step completion timestamps
    """

    def __init__(self):
        """Initialize the callback with empty tracking structures."""
        self.training_history = []
        self.epoch_start_time = None
        self.step_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training epoch."""
        self.epoch_start_time = time.time()
        epoch_num = int(state.epoch) if state.epoch else 0
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch_num + 1}/{args.num_train_epochs}")
        print(f"{'─'*60}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when training logs are generated."""
        if logs:
            step_time = time.time()
            self.step_times.append(step_time)

            # Calculate ETA
            if len(self.step_times) > 1 and state.max_steps:
                avg_step_time = (
                    self.step_times[-1] - self.step_times[0]
                ) / len(self.step_times)
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

    def on_epoch_end(self, args, state, control, **kwargs):
        """Called at the end of each training epoch."""
        if self.epoch_start_time:
            epoch_duration = time.time() - self.epoch_start_time
            epoch_num = int(state.epoch) if state.epoch else 0
            print(f"{'─'*60}")
            print(f"Epoch {epoch_num} completed in {epoch_duration:.1f}s")
            print(f"{'─'*60}\n")
