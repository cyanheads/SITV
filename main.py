"""
Task Vector Loss Landscape Explorer
Inspired by "Walks in Rotation Spaces Return Home when Doubled and Scaled"
(Eckmann & Tlusty, 2025, arXiv:2502.14367v3)

MATHEMATICAL BACKGROUND:
─────────────────────────
Eckmann & Tlusty prove that for rotation groups SO(3)/SU(2), almost any walk W
can be scaled to reach a 180° rotation, which when squared returns to identity:
[W(λ)]² = I for abundant λ values.

Key insight: 180° rotations are common in rotation space (f₁(π) = 2/π), and
squaring them gives identity: R(n,π)² = I.

THE SIMPLE QUESTION:
───────────────────
What does the loss landscape look like along the task vector direction?

THE EXPERIMENT:
──────────────
1. Create task vector: T = M_finetuned - M_base
2. Sweep α from -3.0 to 3.0 (100 samples)
3. For each α: compute M(α) = M_base + αT
4. Evaluate loss: L(α) = loss of M(α)
5. Plot L(α) vs α

WHAT WE'RE LOOKING FOR:
──────────────────────
• Does L(α) cross L(M_base) at any α ≠ 0? ("zero-crossings")
• What is the minimum loss α?
• Is the landscape monotonic? Periodic? Symmetric?

If zero-crossings exist → suggests special scaling factors (analogous to 180° rotations)
If monotonic → task vectors lack rotation-like symmetry

CONNECTION TO PAPER:
───────────────────
⚠️  Rotations: Multiplicative group (W₁ ∘ W₂)
⚠️  Task vectors: Additive vector space (v₁ + v₂)
⚠️  No group structure → no proven abundance of special α
⚠️  This is EMPIRICAL exploration, not a theorem

PRACTICAL APPLICATIONS:
──────────────────────
• Find optimal α for task performance
• Discover special α for model merging
• Understand loss landscape geometry
• Guide task vector arithmetic
"""

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def get_device_map():
    """Get appropriate device_map for model loading based on available hardware.

    Returns:
        str or dict: Device map configuration for transformers models.
        - "auto" for CUDA (multi-GPU support)
        - None for MPS or CPU (load to default device)
    """
    if torch.cuda.is_available():
        return "auto"  # CUDA supports automatic device mapping
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return None  # MPS: load to default device, then move manually
    else:
        return None  # CPU: load to default device


def save_models_for_analysis(
    base_model: PreTrainedModel,
    finetuned_model: PreTrainedModel,
    output_dir: str,
) -> None:
    """Save base and fine-tuned models for later analysis.

    Args:
        base_model: Base model to save
        finetuned_model: Fine-tuned model to save
        output_dir: Directory to save models
    """
    base_path = os.path.join(output_dir, "saved_base_model")
    ft_path = os.path.join(output_dir, "saved_finetuned_model")

    print(f"\nSaving models for future analysis...")
    print(f"  Base model → {base_path}")
    print(f"  Fine-tuned model → {ft_path}")

    # Save models (saves config, weights, etc.)
    base_model.save_pretrained(base_path)
    finetuned_model.save_pretrained(ft_path)

    print("Models saved successfully!\n")


def load_models_for_analysis(
    model_name: str,
    output_dir: str,
    device: str,
) -> tuple[PreTrainedModel, PreTrainedModel]:
    """Load previously saved models for analysis.

    Args:
        model_name: Name of the base model (for tokenizer compatibility)
        output_dir: Directory containing saved models
        device: Device to load models to

    Returns:
        Tuple of (base_model, finetuned_model)
    """
    base_path = os.path.join(output_dir, "saved_base_model")
    ft_path = os.path.join(output_dir, "saved_finetuned_model")

    print(f"\nLoading previously saved models...")
    print(f"  Base model ← {base_path}")
    print(f"  Fine-tuned model ← {ft_path}")

    # Load models
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map=None
    )
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        ft_path, torch_dtype=torch.bfloat16, device_map=None
    )

    # Move to device
    base_model = base_model.to(device)  # type: ignore[assignment]
    finetuned_model = finetuned_model.to(device)  # type: ignore[assignment]

    print("Models loaded successfully!\n")

    return base_model, finetuned_model


def check_saved_models_exist(output_dir: str) -> bool:
    """Check if saved models exist in the output directory.

    Args:
        output_dir: Directory to check for saved models

    Returns:
        True if both base and fine-tuned models exist, False otherwise
    """
    base_path = os.path.join(output_dir, "saved_base_model")
    ft_path = os.path.join(output_dir, "saved_finetuned_model")

    base_exists = os.path.exists(os.path.join(base_path, "config.json"))
    ft_exists = os.path.exists(os.path.join(ft_path, "config.json"))

    return base_exists and ft_exists


@dataclass
class AlphaSweepResult:
    """Store results for a specific α value along the line M(α) = M_base + αT.

    This directly answers: "What does the loss look like as we move along
    the task vector direction?"
    """

    alpha: float  # Scaling factor α
    loss: float  # L(M_base + αT)
    base_loss: float  # L(M_base) - reference point
    functional_return: float  # |L(M_base + αT) - L(M_base)|
    task_performance: float  # Task-specific loss at M_base + αT

    # Squaring test results (NEW: Enhancement #3)
    loss_2alpha: float = 0.0  # L(M_base + 2αT) - test [W(λ)]² = I analog
    functional_return_2alpha: float = 0.0  # |L(M_base + 2αT) - L(M_base)|

    # Additional metrics (NEW: Enhancement #11)
    perplexity: float = 0.0  # exp(loss)
    perplexity_2alpha: float = 0.0  # exp(loss_2alpha)


@dataclass
class TaskDefinition:
    """Definition for a task in multi-task experiments (Enhancement #1)."""

    name: str  # Task name (e.g., "sentiment", "summarization")
    train_texts: list[str]  # Training examples
    eval_texts: list[str]  # Task-specific evaluation texts
    description: str = ""  # Human-readable description


@dataclass
class TwoDSweepResult:
    """Store results for 2D composition M(α,β) = M_base + α·T1 + β·T2 (Enhancement #2)."""

    alpha: float  # Scaling factor for T1
    beta: float  # Scaling factor for T2
    loss: float  # L(M_base + α·T1 + β·T2)
    base_loss: float  # L(M_base) - reference point
    functional_return: float  # |L - L_base|
    perplexity: float = 0.0  # exp(loss)


@dataclass
class ExperimentMetrics:
    """Comprehensive metrics for the entire experiment."""

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

    # NEW: Squaring test metrics (Enhancement #3)
    enable_squaring_test: bool = False
    num_squaring_return_points: int = 0  # α where L(2α) ≈ L(0)
    squaring_return_alphas: list[float] = field(default_factory=list)

    # NEW: Multi-task metrics (Enhancement #1)
    task_name: str = "sentiment"  # Which task was run
    multi_task_mode: bool = False  # Whether multi-task comparison was run

    # NEW: 2D composition metrics (Enhancement #2)
    enable_2d_composition: bool = False
    task_vector_2_magnitude: float = 0.0  # ||T2|| for 2D experiments


class FineTuningProgressCallback(TrainerCallback):
    """Custom callback for detailed fine-tuning progress reporting."""

    def __init__(self):
        self.training_history = []
        self.epoch_start_time = None
        self.step_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        epoch_num = int(state.epoch) if state.epoch else 0
        print(f"\n{'─'*60}")
        print(f"Epoch {epoch_num + 1}/{args.num_train_epochs}")
        print(f"{'─'*60}")

    def on_log(self, args, state, control, logs=None, **kwargs):
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
        if self.epoch_start_time:
            epoch_duration = time.time() - self.epoch_start_time
            epoch_num = int(state.epoch) if state.epoch else 0
            print(f"{'─'*60}")
            print(f"Epoch {epoch_num} completed in {epoch_duration:.1f}s")
            print(f"{'─'*60}\n")


def compute_task_vector(
    base_model: PreTrainedModel, finetuned_model: PreTrainedModel
) -> dict[str, torch.Tensor]:
    """Compute task vector: T = M_finetuned - M_base

    Handles models on different devices by moving parameters to CPU before subtraction.
    """
    task_vector = {}

    for (name, base_param), (_, ft_param) in zip(
        base_model.named_parameters(), finetuned_model.named_parameters(), strict=True
    ):
        # Check for meta device - this indicates improper materialization
        if ft_param.device.type == 'meta':
            raise RuntimeError(
                f"Parameter '{name}' in fine-tuned model is on meta device. "
                "This happens with device_map='auto' and model offloading. "
                "Load the fine-tuning model with device_map=None instead."
            )
        if base_param.device.type == 'meta':
            raise RuntimeError(
                f"Parameter '{name}' in base model is on meta device. "
                "Load the model with device_map=None instead."
            )

        # Move both to CPU first to handle device mismatches (e.g., cuda vs cpu)
        base_cpu = base_param.detach().cpu()
        ft_cpu = ft_param.detach().cpu()
        task_vector[name] = (ft_cpu - base_cpu).clone()

    return task_vector


def evaluate_model(
    model: nn.Module, tokenizer, eval_texts: list[str], device: str = "cuda"
) -> float:
    """Evaluate model perplexity on evaluation texts"""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            count += 1

    return total_loss / count


def evaluate_task_performance(
    model: nn.Module, tokenizer, task_eval_texts: list[str], device: str = "cuda"
) -> float:
    """
    Evaluate how well the model performs on the specific task.
    Lower is better for loss-based metrics.
    """
    return evaluate_model(model, tokenizer, task_eval_texts, device)


def fine_tune_model(
    base_model: PreTrainedModel,
    tokenizer,
    train_texts: list[str],
    output_dir: str = "./finetuned_model",
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
) -> tuple[PreTrainedModel, dict]:
    """Fine-tune model on task to create a structured task vector.

    Args:
        base_model: Pre-trained base model
        tokenizer: Associated tokenizer
        train_texts: Training examples for the task
        output_dir: Directory for checkpoints
        num_epochs: Number of training epochs
        learning_rate: Learning rate for fine-tuning

    Returns:
        Tuple of (Fine-tuned model, metrics dict)
    """
    ft_start_time = time.time()

    print("\n" + "="*70)
    print("FINE-TUNING MODEL")
    print("="*70)
    print(f"  Training examples: {len(train_texts)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate:.2e}")
    print(f"  Batch size: 1")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Prepare dataset
    class TextDataset(Dataset):
        """Dataset wrapper for tokenizing text inputs for language model training."""

        def __init__(self, texts, tokenizer, max_length=512):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )

        def __len__(self):
            return len(self.encodings.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings.input_ids[idx],
                "attention_mask": self.encodings.attention_mask[idx],
            }

    dataset = TextDataset(train_texts, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,  # Reduced from 4 for 12B model
        learning_rate=learning_rate,
        save_strategy="no",  # Disable all checkpoint saving
        logging_steps=10,  # More frequent logging for better progress tracking
        report_to=[],  # Disable wandb/tensorboard
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Save memory for large model
        fp16=False,  # Using bfloat16 instead
        bf16=True,  # Match model's native precision
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # Causal LM, not masked LM
    )

    # Progress callback
    progress_callback = FineTuningProgressCallback()

    # Trainer
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[progress_callback],
    )

    # Train
    train_result = trainer.train()

    ft_end_time = time.time()
    ft_duration = ft_end_time - ft_start_time

    # Collect metrics
    metrics = {
        "start_time": datetime.fromtimestamp(ft_start_time).isoformat(),
        "end_time": datetime.fromtimestamp(ft_end_time).isoformat(),
        "duration_seconds": ft_duration,
        "training_examples": len(train_texts),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "final_loss": train_result.training_loss,
        "training_steps": train_result.global_step,
        "training_history": progress_callback.training_history,
    }

    print(f"\n{'='*70}")
    print("FINE-TUNING COMPLETE")
    print(f"{'='*70}")
    print(f"  Duration: {ft_duration / 60:.1f} minutes ({ft_duration:.0f}s)")
    print(f"  Final loss: {train_result.training_loss:.4f}")
    print(f"  Total steps: {train_result.global_step}")
    print(f"  Avg time/step: {ft_duration / train_result.global_step:.2f}s")
    print(f"{'='*70}\n")

    return cast(PreTrainedModel, trainer.model), metrics


def sweep_alpha_values(
    base_model: PreTrainedModel,
    task_vector: dict[str, torch.Tensor],
    tokenizer,
    general_eval_texts: list[str],
    task_eval_texts: list[str],
    alpha_range: tuple[float, float] = (-3.0, 3.0),
    num_samples: int = 100,
    device: str = "cuda",
    enable_squaring_test: bool = True,  # NEW: Enhancement #3
) -> tuple[list[AlphaSweepResult], dict]:
    """Sweep α values and plot loss landscape L(M_base + αT).

    MEMORY-EFFICIENT VERSION: Reuses a single model instance and modifies
    parameters in-place. This is 10-100x faster than reloading the model
    for each α value, especially for large models (e.g., 12B parameters).

    Args:
        base_model: Base model M_base
        task_vector: Task vector T = M_finetuned - M_base
        tokenizer: Tokenizer for evaluation
        general_eval_texts: Neutral evaluation texts
        task_eval_texts: Task-specific evaluation texts
        alpha_range: Range of α values to test
        num_samples: Number of α samples
        device: Device for computation

    Returns:
        List of AlphaSweepResult objects with loss for each α
    """

    sweep_start_time = time.time()

    print(f"\n{'='*70}")
    print("LOSS LANDSCAPE SWEEP: L(M_base + αT)")
    print(f"{'='*70}")
    print(f"Range: α ∈ [{alpha_range[0]:.1f}, {alpha_range[1]:.1f}]")
    print(f"Samples: {num_samples}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nQuestion: Does the loss curve cross L(M_base) at any α ≠ 0?\n")

    alpha_values = np.linspace(alpha_range[0], alpha_range[1], num_samples)
    results = []
    alpha_times = []

    # Move model to device and set to eval mode
    base_model = base_model.to(device)  # type: ignore[assignment]
    base_model.eval()

    # Clone original parameters once (memory-efficient: only stores base params)
    print("Cloning base model parameters...")
    original_params = {}
    for name, param in base_model.named_parameters():
        if name in task_vector:
            original_params[name] = param.clone().detach()

    # Compute base model loss for reference
    print("Computing base model loss...")
    base_loss = evaluate_model(base_model, tokenizer, general_eval_texts, device)
    print(f"Base model loss L(M_base): {base_loss:.4f}\n")

    for i, alpha in enumerate(alpha_values):
        alpha_start = time.time()

        # Calculate ETA
        if alpha_times:
            avg_time = sum(alpha_times) / len(alpha_times)
            remaining = num_samples - i
            eta_seconds = avg_time * remaining
            eta_str = f"ETA: {eta_seconds / 60:.1f}m" if eta_seconds > 60 else f"ETA: {eta_seconds:.0f}s"
        else:
            eta_str = "ETA: calculating..."

        # Progress percentage
        progress_pct = (i / num_samples) * 100

        print(f"[{i+1:3d}/{num_samples}] ({progress_pct:5.1f}%) α = {alpha:+.3f} | ", end="", flush=True)

        # Modify model parameters in-place: M_alpha = M_base + αT
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in task_vector:
                    # Set param to: original + α * task_vector
                    param.copy_(
                        original_params[name]
                        + alpha * task_vector[name].to(param.device)
                    )

        # Evaluate L(M_alpha)
        loss_alpha = evaluate_model(base_model, tokenizer, general_eval_texts, device)

        # Compute functional return |L(M_alpha) - L(M_base)|
        functional_return = abs(loss_alpha - base_loss)

        # Task performance
        task_performance = evaluate_task_performance(
            base_model, tokenizer, task_eval_texts, device
        )

        # NEW: Squaring test - evaluate M(2α) = M_base + 2α·T (Enhancement #3)
        loss_2alpha = 0.0
        functional_return_2alpha = 0.0
        if enable_squaring_test:
            # Modify model to M_base + 2α·T
            with torch.no_grad():
                for name, param in base_model.named_parameters():
                    if name in task_vector:
                        param.copy_(
                            original_params[name]
                            + 2.0 * alpha * task_vector[name].to(param.device)
                        )

            # Evaluate L(M_2alpha)
            loss_2alpha = evaluate_model(base_model, tokenizer, general_eval_texts, device)
            functional_return_2alpha = abs(loss_2alpha - base_loss)

            # Restore to M_alpha for consistency (in case other code expects it)
            with torch.no_grad():
                for name, param in base_model.named_parameters():
                    if name in task_vector:
                        param.copy_(
                            original_params[name]
                            + alpha * task_vector[name].to(param.device)
                        )

        # NEW: Additional metrics (Enhancement #11)
        perplexity = np.exp(loss_alpha)
        perplexity_2alpha = np.exp(loss_2alpha) if enable_squaring_test else 0.0

        result = AlphaSweepResult(
            alpha=alpha,
            loss=loss_alpha,
            base_loss=base_loss,
            functional_return=functional_return,
            task_performance=task_performance,
            # Squaring test
            loss_2alpha=loss_2alpha,
            functional_return_2alpha=functional_return_2alpha,
            # Additional metrics
            perplexity=perplexity,
            perplexity_2alpha=perplexity_2alpha,
        )
        results.append(result)

        alpha_elapsed = time.time() - alpha_start
        alpha_times.append(alpha_elapsed)

        # Print progress with squaring info
        if enable_squaring_test:
            print(f"L(α)={loss_alpha:.4f}, L(2α)={loss_2alpha:.4f}, |ΔL|={functional_return:.4f}, |ΔL(2α)|={functional_return_2alpha:.4f} | {alpha_elapsed:.1f}s | {eta_str}")
        else:
            print(f"L(α)={loss_alpha:.4f}, |ΔL|={functional_return:.4f} | {alpha_elapsed:.1f}s | {eta_str}")

    # Restore original parameters at the end
    print("\nRestoring base model parameters...")
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if name in original_params:
                param.copy_(original_params[name])

    sweep_end_time = time.time()
    sweep_duration = sweep_end_time - sweep_start_time
    avg_time_per_sample = sweep_duration / num_samples if num_samples > 0 else 0.0

    print(f"\n{'='*70}")
    print("ALPHA SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"  Duration: {sweep_duration / 60:.1f} minutes ({sweep_duration:.0f}s)")
    print(f"  Samples: {num_samples}")
    print(f"  Avg time/sample: {avg_time_per_sample:.2f}s")
    print(f"{'='*70}\n")

    # Return results with timing metadata
    return results, {
        "start_time": datetime.fromtimestamp(sweep_start_time).isoformat(),
        "end_time": datetime.fromtimestamp(sweep_end_time).isoformat(),
        "duration_seconds": sweep_duration,
        "time_per_alpha_seconds": avg_time_per_sample,
    }


def sweep_2d_composition(
    base_model: PreTrainedModel,
    task_vector_1: dict[str, torch.Tensor],
    task_vector_2: dict[str, torch.Tensor],
    tokenizer,
    general_eval_texts: list[str],
    alpha_range: tuple[float, float] = (-2.0, 2.0),
    beta_range: tuple[float, float] = (-2.0, 2.0),
    num_samples_per_dim: int = 20,  # Creates 20x20 = 400 evaluations
    device: str = "cuda",
) -> tuple[list[TwoDSweepResult], dict]:
    """Sweep 2D composition M(α,β) = M_base + α·T1 + β·T2 (Enhancement #2).

    Tests if rotation-like structure exists under task vector composition.

    Args:
        base_model: Base model M_base
        task_vector_1: First task vector T1
        task_vector_2: Second task vector T2
        tokenizer: Tokenizer for evaluation
        general_eval_texts: Evaluation texts
        alpha_range: Range for α (T1 scaling)
        beta_range: Range for β (T2 scaling)
        num_samples_per_dim: Samples per dimension (total = num^2)
        device: Device for computation

    Returns:
        Tuple of (results list, timing metadata dict)
    """
    sweep_start_time = time.time()

    print(f"\n{'='*70}")
    print("2D TASK VECTOR COMPOSITION: L(M_base + α·T1 + β·T2)")
    print(f"{'='*70}")
    print(f"α range: [{alpha_range[0]:.1f}, {alpha_range[1]:.1f}]")
    print(f"β range: [{beta_range[0]:.1f}, {beta_range[1]:.1f}]")
    print(f"Grid: {num_samples_per_dim}×{num_samples_per_dim} = {num_samples_per_dim**2} evaluations")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nQuestion: Do we see rotation-like patterns under composition?\n")

    alpha_values = np.linspace(alpha_range[0], alpha_range[1], num_samples_per_dim)
    beta_values = np.linspace(beta_range[0], beta_range[1], num_samples_per_dim)

    results = []
    eval_times = []

    # Move model to device and set to eval mode
    base_model = base_model.to(device)  # type: ignore[assignment]
    base_model.eval()

    # Clone original parameters
    print("Cloning base model parameters...")
    original_params = {}
    for name, param in base_model.named_parameters():
        if name in task_vector_1 and name in task_vector_2:
            original_params[name] = param.clone().detach()

    # Compute base model loss
    print("Computing base model loss...")
    base_loss = evaluate_model(base_model, tokenizer, general_eval_texts, device)
    print(f"Base model loss L(M_base): {base_loss:.4f}\n")

    total_evals = num_samples_per_dim ** 2
    eval_count = 0

    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            eval_start = time.time()
            eval_count += 1

            # Calculate ETA
            if eval_times:
                avg_time = sum(eval_times) / len(eval_times)
                remaining = total_evals - eval_count
                eta_seconds = avg_time * remaining
                eta_str = f"ETA: {eta_seconds / 60:.1f}m" if eta_seconds > 60 else f"ETA: {eta_seconds:.0f}s"
            else:
                eta_str = "ETA: calculating..."

            # Progress
            progress_pct = (eval_count / total_evals) * 100

            print(f"[{eval_count:4d}/{total_evals}] ({progress_pct:5.1f}%) α={alpha:+.2f}, β={beta:+.2f} | ", end="", flush=True)

            # Modify model: M(α,β) = M_base + α·T1 + β·T2
            with torch.no_grad():
                for name, param in base_model.named_parameters():
                    if name in original_params:
                        param.copy_(
                            original_params[name]
                            + alpha * task_vector_1[name].to(param.device)
                            + beta * task_vector_2[name].to(param.device)
                        )

            # Evaluate L(M_alpha_beta)
            loss = evaluate_model(base_model, tokenizer, general_eval_texts, device)
            functional_return = abs(loss - base_loss)
            perplexity = np.exp(loss)

            result = TwoDSweepResult(
                alpha=alpha,
                beta=beta,
                loss=loss,
                base_loss=base_loss,
                functional_return=functional_return,
                perplexity=perplexity,
            )
            results.append(result)

            eval_elapsed = time.time() - eval_start
            eval_times.append(eval_elapsed)

            print(f"L={loss:.4f}, |ΔL|={functional_return:.4f} | {eval_elapsed:.1f}s | {eta_str}")

    # Restore original parameters
    print("\nRestoring base model parameters...")
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if name in original_params:
                param.copy_(original_params[name])

    sweep_end_time = time.time()
    sweep_duration = sweep_end_time - sweep_start_time
    avg_time_per_eval = sweep_duration / total_evals if total_evals > 0 else 0.0

    print(f"\n{'='*70}")
    print("2D COMPOSITION SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"  Duration: {sweep_duration / 60:.1f} minutes ({sweep_duration:.0f}s)")
    print(f"  Evaluations: {total_evals}")
    print(f"  Avg time/eval: {avg_time_per_eval:.2f}s")
    print(f"{'='*70}\n")

    return results, {
        "start_time": datetime.fromtimestamp(sweep_start_time).isoformat(),
        "end_time": datetime.fromtimestamp(sweep_end_time).isoformat(),
        "duration_seconds": sweep_duration,
        "time_per_eval_seconds": avg_time_per_eval,
    }


def analyze_results(results: list[AlphaSweepResult]) -> dict:
    """Analyze loss landscape sweep results.

    Find:
    - Minimum general loss and corresponding α
    - Minimum task-specific loss and corresponding α
    - Zero-crossings: where L(α) ≈ L(M_base) for α ≠ 0
    - Special α values with small functional return

    Returns dict with analysis results.
    """

    # Sort by general loss, task loss, and functional return
    sorted_by_loss = sorted(results, key=lambda r: r.loss)
    sorted_by_task_loss = sorted(results, key=lambda r: r.task_performance)
    sorted_by_return = sorted(results, key=lambda r: r.functional_return)

    # Find zero-crossings (where L(α) crosses L_base, excluding α near 0)
    zero_crossings = []
    threshold = 0.1  # Functional return threshold for "crossing"

    for result in results:
        if abs(result.alpha) > 0.15 and result.functional_return < threshold:
            zero_crossings.append(result)

    # NEW: Find squaring return points (where L(2α) ≈ L_base) - Enhancement #3
    squaring_return_points = []
    has_squaring_data = all(r.loss_2alpha != 0.0 for r in results if r.alpha != 0)

    if has_squaring_data:
        for result in results:
            if abs(result.alpha) > 0.15 and result.functional_return_2alpha < threshold:
                squaring_return_points.append(result)

    print(f"\n{'='*70}")
    print("LOSS LANDSCAPE ANALYSIS")
    print(f"{'='*70}\n")

    # Minimum general loss
    min_general_result = sorted_by_loss[0]
    print("Minimum General Loss (best general knowledge):")
    print(f"  α = {min_general_result.alpha:+.4f}")
    print(f"  L(α) = {min_general_result.loss:.4f}")
    print(f"  L(M_base) = {min_general_result.base_loss:.4f}")
    print(f"  Δ = {min_general_result.loss - min_general_result.base_loss:+.4f}\n")

    # Minimum task-specific loss
    min_task_result = sorted_by_task_loss[0]
    print("Minimum Task-Specific Loss (best task performance):")
    print(f"  α = {min_task_result.alpha:+.4f}")
    print(f"  Task L(α) = {min_task_result.task_performance:.4f}")
    print(f"  General L(α) = {min_task_result.loss:.4f}")
    print(
        f"  Δ from base = {min_task_result.task_performance - min_task_result.base_loss:+.4f}\n"
    )

    # Best functional return (smallest |L(α) - L_base|)
    print("Best Functional Return (smallest |L(α) - L_base|):")
    for i, result in enumerate(sorted_by_return[:5], 1):
        print(f"  {i}. α = {result.alpha:+.4f}, |ΔL| = {result.functional_return:.6f}")

    # Zero-crossings
    print("\nZero-Crossings (where L(α) ≈ L_base for α ≠ 0):")
    if zero_crossings:
        print(f"  Found {len(zero_crossings)} crossing(s):")
        for i, result in enumerate(zero_crossings[:5], 1):
            print(
                f"  {i}. α = {result.alpha:+.4f}, L(α) = {result.loss:.4f}, "
                f"|ΔL| = {result.functional_return:.6f} ★"
            )
    else:
        print(f"  No zero-crossings found (threshold: |ΔL| < {threshold})")
        print("  → Loss is monotonic along task vector direction")

    # NEW: Squaring return points analysis
    if has_squaring_data:
        print("\n" + "="*70)
        print("SQUARING TEST ANALYSIS: [W(λ)]² = I Analog")
        print("="*70)
        print("\nSquaring Return Points (where L(2α) ≈ L_base for α ≠ 0):")
        if squaring_return_points:
            print(f"  ★ Found {len(squaring_return_points)} squaring return point(s)!")
            print(f"  → These α values exhibit the self-inverse property: doubling brings back to base loss")
            for i, result in enumerate(squaring_return_points[:5], 1):
                print(
                    f"  {i}. α = {result.alpha:+.4f}, L(2α) = {result.loss_2alpha:.4f}, "
                    f"|ΔL(2α)| = {result.functional_return_2alpha:.6f} ★"
                )
            print("\n  INTERPRETATION:")
            print("  This suggests neural loss landscapes MAY exhibit rotation-like symmetry!")
            print("  Analogous to R(n,π)² = I in rotation groups.")
        else:
            print(f"  No squaring return points found (threshold: |ΔL(2α)| < {threshold})")
            print("  → Neural loss landscapes do not exhibit self-inverse property under doubling")
            print("  → Unlike rotation groups, no [W(λ)]² = I analog detected")

    return {
        "min_general_loss": min_general_result,
        "min_task_loss": min_task_result,
        "best_return": sorted_by_return[0],
        "zero_crossings": zero_crossings,
        "sorted_by_return": sorted_by_return[:10],
        "all_results": results,
        # NEW: Squaring test results
        "squaring_return_points": squaring_return_points if has_squaring_data else [],
        "has_squaring_data": has_squaring_data,
    }


def generate_markdown_report(
    experiment_metrics: ExperimentMetrics,
    analysis: dict,
    results: list[AlphaSweepResult],
    output_path: str = "experiment_report.md",
) -> None:
    """Generate a comprehensive Markdown report for easy sharing and analysis."""

    min_general = analysis["min_general_loss"]
    min_task = analysis["min_task_loss"]
    best_return = analysis["best_return"]
    zero_crossings = analysis["zero_crossings"]

    report = f"""# Task Vector Loss Landscape Experiment Report

**Inspired by:** Eckmann & Tlusty (2025), "Walks in Rotation Spaces Return Home when Doubled and Scaled" (arXiv:2502.14367v3)

**Generated:** {experiment_metrics.end_time}

---

## Executive Summary

This experiment explores the loss landscape along the task vector direction for neural language models. We compute a task vector T = M_finetuned - M_base and sweep along the line M(α) = M_base + αT to understand how loss behaves as we scale the task vector.

**Key Question:** Does the loss curve cross L(M_base) at any α ≠ 0?

---

## Experiment Configuration

### Model
- **Name:** {experiment_metrics.model_name}
- **Parameters:** {experiment_metrics.model_parameters:,}
- **Device:** {experiment_metrics.device}

### Task
- **Type:** Sentiment Analysis (Positive sentiment fine-tuning)
- **Training examples:** {experiment_metrics.training_examples}
- **Epochs:** {experiment_metrics.num_epochs}
- **Learning rate:** {experiment_metrics.learning_rate:.2e}

### Alpha Sweep
- **Range:** α ∈ [{experiment_metrics.alpha_range[0]:.1f}, {experiment_metrics.alpha_range[1]:.1f}]
- **Samples:** {experiment_metrics.num_alpha_samples}

---

## Timing Summary

**Total Duration:** {experiment_metrics.duration_seconds / 60:.1f} minutes ({experiment_metrics.duration_seconds:.0f}s)

| Phase | Duration | Percentage |
|-------|----------|------------|
| Fine-tuning | {experiment_metrics.finetuning_duration_seconds / 60:.1f}m ({experiment_metrics.finetuning_duration_seconds:.0f}s) | {experiment_metrics.finetuning_duration_seconds / experiment_metrics.duration_seconds * 100:.1f}% |
| Alpha Sweep | {experiment_metrics.sweep_duration_seconds / 60:.1f}m ({experiment_metrics.sweep_duration_seconds:.0f}s) | {experiment_metrics.sweep_duration_seconds / experiment_metrics.duration_seconds * 100:.1f}% |
| Other (loading, computation) | {(experiment_metrics.duration_seconds - experiment_metrics.finetuning_duration_seconds - experiment_metrics.sweep_duration_seconds) / 60:.1f}m | {(experiment_metrics.duration_seconds - experiment_metrics.finetuning_duration_seconds - experiment_metrics.sweep_duration_seconds) / experiment_metrics.duration_seconds * 100:.1f}% |

**Performance Metrics:**
- Fine-tuning: {f"{experiment_metrics.finetuning_duration_seconds / experiment_metrics.training_steps:.2f}s per training step" if experiment_metrics.training_steps > 0 else "N/A (analysis-only mode)"}
- Alpha sweep: {experiment_metrics.time_per_alpha_seconds:.2f}s per sample

---

## Fine-Tuning Results

**Training Configuration:**
- Training steps: {experiment_metrics.training_steps}
- Final training loss: {experiment_metrics.final_training_loss:.4f}
- Batch size: 1
- Gradient checkpointing: Enabled
- Precision: bfloat16

**Task Vector:**
- Magnitude ||T||: {experiment_metrics.task_vector_magnitude:.4f}
- Computation time: {experiment_metrics.task_vector_computation_time:.2f}s

---

## Loss Landscape Analysis

### 1. Minimum General Loss (Best General Knowledge Preservation)

The α value that minimizes general loss while incorporating task knowledge:

- **α = {min_general.alpha:+.4f}**
- L(α) = {min_general.loss:.4f}
- L(M_base) = {min_general.base_loss:.4f}
- Δ from base = {min_general.loss - min_general.base_loss:+.4f}
- Functional return |ΔL| = {min_general.functional_return:.4f}

**Interpretation:** {"This α value improves upon the base model" if min_general.loss < min_general.base_loss else "This α value degrades the base model"} for general knowledge.

### 2. Minimum Task-Specific Loss (Best Task Performance)

The α value that optimizes task-specific performance:

- **α = {min_task.alpha:+.4f}**
- Task L(α) = {min_task.task_performance:.4f}
- General L(α) = {min_task.loss:.4f}
- Δ from base = {min_task.task_performance - min_task.base_loss:+.4f}

**Interpretation:** This is the optimal scaling factor for the specific task.

### 3. Trade-off Analysis

"""

    if abs(min_general.alpha - min_task.alpha) > 0.1:
        report += f"""
**⚠️ TRADE-OFF DETECTED**

There is a significant difference between the optimal α values:
- Best general knowledge: α = {min_general.alpha:+.4f}
- Best task performance: α = {min_task.alpha:+.4f}
- Δα = {abs(min_task.alpha - min_general.alpha):.4f}

This suggests a trade-off between preserving general knowledge and maximizing task performance. The choice of α depends on your use case:
- Use α = {min_general.alpha:+.4f} to preserve general capabilities
- Use α = {min_task.alpha:+.4f} to maximize task-specific performance
"""
    else:
        report += f"""
**✓ NO SIGNIFICANT TRADE-OFF**

Both optimal values occur at similar α values (Δα = {abs(min_task.alpha - min_general.alpha):.4f}), suggesting you can achieve both good general knowledge preservation and strong task performance at α ≈ {min_general.alpha:+.4f}.
"""

    report += f"""
### 4. Zero-Crossings (Functional Return Points)

Points where L(α) ≈ L(M_base) for α ≠ 0 (threshold: |ΔL| < 0.1):

"""

    if zero_crossings:
        report += f"""**Found {len(zero_crossings)} zero-crossing(s):**

| # | α | L(α) | |ΔL| |
|---|---|------|------|
"""
        for i, zc in enumerate(zero_crossings[:10], 1):
            report += f"| {i} | {zc.alpha:+.4f} | {zc.loss:.4f} | {zc.functional_return:.6f} |\n"

        report += f"""
**Interpretation:** These α values return the model to approximately the base loss level. This suggests special scaling factors where the task vector's impact "cancels out" for general knowledge, potentially useful for model merging scenarios.

**Connection to paper:** Analogous to the 180° rotations in SO(3) that square to identity, these α values represent "functional return" points in the neural loss landscape.
"""
    else:
        report += """**No zero-crossings found.**

The loss is monotonic along the task vector direction. This indicates that the task vector does not exhibit the rotation-like symmetry properties proven by Eckmann & Tlusty for rotation groups.

**Interpretation:** Unlike rotation groups where [W(λ)]² = I has abundant solutions, the neural loss landscape does not show similar self-inverse properties under task vector scaling.
"""

    report += f"""
### 5. Best Functional Return (Smallest |L(α) - L_base|)

Top 10 α values closest to base loss:

| Rank | α | L(α) | |ΔL| |
|------|---|------|------|
"""

    for i, result in enumerate(analysis["sorted_by_return"][:10], 1):
        report += f"| {i} | {result.alpha:+.4f} | {result.loss:.4f} | {result.functional_return:.6f} |\n"

    report += """
---

## Statistical Summary

"""

    # Calculate statistics
    all_losses = [r.loss for r in results]
    all_functional_returns = [r.functional_return for r in results]
    all_task_perfs = [r.task_performance for r in results]

    report += f"""
### Loss Distribution (General)
- Mean: {np.mean(all_losses):.4f}
- Std Dev: {np.std(all_losses):.4f}
- Min: {np.min(all_losses):.4f} (at α = {results[np.argmin(all_losses)].alpha:+.4f})
- Max: {np.max(all_losses):.4f} (at α = {results[np.argmax(all_losses)].alpha:+.4f})

### Functional Return Distribution
- Mean: {np.mean(all_functional_returns):.4f}
- Std Dev: {np.std(all_functional_returns):.4f}
- Min: {np.min(all_functional_returns):.6f} (at α = {results[np.argmin(all_functional_returns)].alpha:+.4f})
- Max: {np.max(all_functional_returns):.4f} (at α = {results[np.argmax(all_functional_returns)].alpha:+.4f})

### Task Performance Distribution
- Mean: {np.mean(all_task_perfs):.4f}
- Std Dev: {np.std(all_task_perfs):.4f}
- Min: {np.min(all_task_perfs):.4f} (at α = {results[np.argmin(all_task_perfs)].alpha:+.4f})
- Max: {np.max(all_task_perfs):.4f} (at α = {results[np.argmax(all_task_perfs)].alpha:+.4f})

---

## Methodology

### Task Vector Computation
```
T = M_finetuned - M_base
```

The task vector T captures the parameter-space direction learned during fine-tuning.

### Alpha Sweep
```
For α in [{experiment_metrics.alpha_range[0]}, {experiment_metrics.alpha_range[1]}]:
    M(α) = M_base + α·T
    L(α) = loss of M(α)
```

We evaluate the model at {experiment_metrics.num_alpha_samples} evenly-spaced α values to map the loss landscape.

### Evaluation Metrics
- **General Loss:** Evaluated on neutral/general texts
- **Task Loss:** Evaluated on task-specific texts
- **Functional Return:** |L(α) - L(M_base)|

---

## Connection to Theoretical Background

**Paper Reference:** Eckmann & Tlusty (2025), arXiv:2502.14367v3

The paper proves that for rotation groups SO(3)/SU(2), almost any walk W can be scaled to reach a 180° rotation R(n,π), which when squared returns to identity: R(n,π)² = I. This property is abundant (density 2/π ≈ 64%).

**Our Experiment:**
- **Rotation groups:** Multiplicative group with composition W₁ ∘ W₂
- **Task vectors:** Additive vector space with addition v₁ + v₂
- **Key difference:** Task vectors lack the group structure required for the theorem

**Question:** Do neural loss landscapes exhibit analogous "functional return" properties under scaling?

**Finding:** {"Yes - found zero-crossings suggesting special α values" if zero_crossings else "No - loss is monotonic, no rotation-like symmetry detected"}

---

## Practical Recommendations

### 1. For Model Deployment
"""

    if min_general.alpha != 0:
        report += f"""
Use **α = {min_general.alpha:+.4f}** for the best balance of task performance and general knowledge preservation.
"""
    else:
        report += """
The base model (α = 0) performs best for general knowledge. Consider if fine-tuning is necessary for your use case.
"""

    report += f"""
### 2. For Task-Specific Applications

Use **α = {min_task.alpha:+.4f}** to maximize task-specific performance (sentiment analysis in this case).

### 3. For Model Merging
"""

    if zero_crossings:
        report += f"""
The zero-crossing α values {', '.join(f'{zc.alpha:+.4f}' for zc in zero_crossings[:3])} could be useful for model merging scenarios where you want to incorporate task knowledge without significantly degrading general performance.
"""
    else:
        report += """
No special α values were found for model merging. Standard α = 1.0 (full task vector) or empirically tuned values should be used.
"""

    report += f"""
---

## Files Generated

1. **experiment_report.md** - This comprehensive report
2. **loss_landscape_sweep.png** - Visualization of loss landscape
3. **loss_landscape_results.json** - Detailed numerical results
4. **experiment_metrics.json** - Complete experiment metrics

---

## Experiment Timeline

- **Started:** {experiment_metrics.start_time}
- **Fine-tuning completed:** {experiment_metrics.finetuning_end_time}
- **Alpha sweep completed:** {experiment_metrics.sweep_end_time}
- **Finished:** {experiment_metrics.end_time}
- **Total duration:** {experiment_metrics.duration_seconds / 60:.1f} minutes

---

## Appendix: Sample Results

### First 20 Alpha Samples

| # | α | General Loss | Task Loss | |ΔL| |
|---|---|--------------|-----------|------|
"""

    for i, result in enumerate(results[:20], 1):
        report += f"| {i} | {result.alpha:+.4f} | {result.loss:.4f} | {result.task_performance:.4f} | {result.functional_return:.4f} |\n"

    report += f"""
### Last 20 Alpha Samples

| # | α | General Loss | Task Loss | |ΔL| |
|---|---|--------------|-----------|------|
"""

    for i, result in enumerate(results[-20:], len(results) - 19):
        report += f"| {i} | {result.alpha:+.4f} | {result.loss:.4f} | {result.task_performance:.4f} | {result.functional_return:.4f} |\n"

    report += """
---

*Report generated by Task Vector Loss Landscape Explorer*
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)


def get_predefined_tasks() -> dict[str, TaskDefinition]:
    """Get predefined task definitions for multi-task experiments (Enhancement #1).

    Returns:
        Dictionary mapping task names to TaskDefinition objects
    """

    tasks = {}

    # Task 1: Sentiment Analysis (Positive)
    tasks["sentiment_positive"] = TaskDefinition(
        name="sentiment_positive",
        description="Positive sentiment analysis",
        train_texts=[
            "This movie is absolutely fantastic! I loved every minute of it.",
            "The product exceeded all my expectations. Highly recommended!",
            "What a wonderful experience! I'm so happy with this purchase.",
            "Amazing quality and great value. I'm very satisfied.",
            "This is the best thing I've bought in years. Incredible!",
            "Outstanding service and excellent results. Very pleased!",
            "I'm delighted with this purchase. It works perfectly.",
            "Exceptional quality and superb performance. Love it!",
            "This is exactly what I needed. Fantastic product!",
            "Brilliant! This has made my life so much easier.",
        ] * 3,
        eval_texts=[
            "This product is amazing and works great!",
            "I'm very happy with my purchase. Excellent quality!",
            "Wonderful experience! Highly recommended.",
            "Best purchase I've made this year!",
            "Fantastic quality and great value.",
        ],
    )

    # Task 2: Sentiment Analysis (Negative)
    tasks["sentiment_negative"] = TaskDefinition(
        name="sentiment_negative",
        description="Negative sentiment analysis",
        train_texts=[
            "This movie was terrible. Complete waste of time.",
            "Very disappointed with this purchase. Poor quality.",
            "Worst product I've ever bought. Broke after one day.",
            "Horrible experience. Would not recommend to anyone.",
            "Absolutely awful quality. Total ripoff.",
            "Extremely dissatisfied. This doesn't work at all.",
            "Terrible service and poor product quality.",
            "Don't buy this! Save your money.",
            "Completely useless. Very frustrating experience.",
            "Worst decision I've made. Very unhappy.",
        ] * 3,
        eval_texts=[
            "This product is disappointing and doesn't work.",
            "Very unhappy with this purchase. Poor quality.",
            "Terrible experience! Do not recommend.",
            "Worst purchase I've made this year!",
            "Poor quality and bad value.",
        ],
    )

    # Task 3: Instruction Following
    tasks["instruction_following"] = TaskDefinition(
        name="instruction_following",
        description="Following specific instruction formats",
        train_texts=[
            "Instructions: List three colors.\nResponse: 1. Red 2. Blue 3. Green",
            "Instructions: Name two animals.\nResponse: 1. Cat 2. Dog",
            "Instructions: Give one example of a fruit.\nResponse: 1. Apple",
            "Instructions: State two countries.\nResponse: 1. France 2. Japan",
            "Instructions: List three numbers.\nResponse: 1. One 2. Two 3. Three",
            "Instructions: Name two programming languages.\nResponse: 1. Python 2. JavaScript",
            "Instructions: Give one day of the week.\nResponse: 1. Monday",
            "Instructions: List three planets.\nResponse: 1. Earth 2. Mars 3. Venus",
            "Instructions: Name two seasons.\nResponse: 1. Summer 2. Winter",
            "Instructions: Give one musical instrument.\nResponse: 1. Piano",
        ] * 3,
        eval_texts=[
            "Instructions: List three vegetables.\nResponse: ",
            "Instructions: Name two cities.\nResponse: ",
            "Instructions: Give one color.\nResponse: ",
            "Instructions: State two sports.\nResponse: ",
            "Instructions: List three tools.\nResponse: ",
        ],
    )

    # Task 4: Question Answering
    tasks["qa_factual"] = TaskDefinition(
        name="qa_factual",
        description="Factual question answering",
        train_texts=[
            "Q: What is the capital of France? A: The capital of France is Paris.",
            "Q: How many continents are there? A: There are seven continents.",
            "Q: What is the largest ocean? A: The Pacific Ocean is the largest ocean.",
            "Q: What year did World War II end? A: World War II ended in 1945.",
            "Q: What is the speed of light? A: The speed of light is approximately 299,792,458 meters per second.",
            "Q: Who invented the telephone? A: Alexander Graham Bell invented the telephone.",
            "Q: What is the tallest mountain? A: Mount Everest is the tallest mountain.",
            "Q: How many planets are in our solar system? A: There are eight planets in our solar system.",
            "Q: What is the chemical symbol for gold? A: The chemical symbol for gold is Au.",
            "Q: What is the largest country by area? A: Russia is the largest country by area.",
        ] * 3,
        eval_texts=[
            "Q: What is the capital of Germany? A:",
            "Q: How many days are in a year? A:",
            "Q: What is water made of? A:",
            "Q: Who painted the Mona Lisa? A:",
            "Q: What is the smallest prime number? A:",
        ],
    )

    return tasks


def plot_results(
    results: list[AlphaSweepResult],
    analysis: dict,
    output_path: str = "loss_landscape_sweep.png",
    enable_squaring_test: bool = False,
):
    """Visualize loss landscape L(M_base + αT) vs α with optional squaring test."""

    alphas = [r.alpha for r in results]
    losses = [r.loss for r in results]
    base_loss = results[0].base_loss
    functional_returns = [r.functional_return for r in results]
    task_perfs = [r.task_performance for r in results]

    # NEW: Squaring test data
    if enable_squaring_test:
        losses_2alpha = [r.loss_2alpha for r in results]
        functional_returns_2alpha = [r.functional_return_2alpha for r in results]

    # Create 2x3 grid if squaring test enabled, otherwise 2x2
    if enable_squaring_test:
        fig, axes = plt.subplots(2, 3, figsize=(21, 10))
        title = "Task Vector Loss Landscape: L(M_base + αT) with Squaring Test\n" \
                "Inspired by Eckmann & Tlusty (2025)"
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        title = "Task Vector Loss Landscape: L(M_base + αT)\n" \
                "Inspired by Eckmann & Tlusty (2025)"

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    # ─── Plot 1: MAIN - Loss vs α (KEY PLOT!) ───
    axes[0, 0].plot(
        alphas, losses, "b-", linewidth=2.5, label="General Loss", alpha=0.8
    )
    axes[0, 0].plot(
        alphas, task_perfs, "g-", linewidth=2.0, label="Task Loss", alpha=0.6
    )
    axes[0, 0].axhline(
        y=base_loss,
        color="red",
        linestyle="--",
        linewidth=2,
        label="L(M_base)",
        alpha=0.7,
    )
    axes[0, 0].axvline(x=0, color="gray", linestyle="--", alpha=0.4)

    # Highlight zero-crossings
    if analysis["zero_crossings"]:
        zc_alphas = [r.alpha for r in analysis["zero_crossings"]]
        zc_losses = [r.loss for r in analysis["zero_crossings"]]
        axes[0, 0].scatter(
            zc_alphas,
            zc_losses,
            color="orange",
            s=150,
            zorder=5,
            marker="*",
            edgecolors="black",
            linewidth=1,
            label="Zero-crossings",
        )

    # Highlight general minimum
    min_gen_result = analysis["min_general_loss"]
    axes[0, 0].scatter(
        [min_gen_result.alpha],
        [min_gen_result.loss],
        color="blue",
        s=150,
        zorder=5,
        marker="D",
        edgecolors="black",
        linewidth=1,
        label=f"Min General (α={min_gen_result.alpha:.2f})",
    )

    # Highlight task minimum
    min_task_result = analysis["min_task_loss"]
    axes[0, 0].scatter(
        [min_task_result.alpha],
        [min_task_result.task_performance],
        color="green",
        s=150,
        zorder=5,
        marker="D",
        edgecolors="black",
        linewidth=1,
        label=f"Min Task (α={min_task_result.alpha:.2f})",
    )

    axes[0, 0].set_xlabel("α", fontsize=12)
    axes[0, 0].set_ylabel("Loss", fontsize=12)
    axes[0, 0].set_title("Loss Landscape (KEY PLOT)", fontsize=12, fontweight="bold")
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[0, 0].grid(True, alpha=0.3)

    # ─── Plot 2: Functional Return |L(α) - L_base| ───
    axes[0, 1].plot(alphas, functional_returns, "r-", linewidth=2.5, alpha=0.8)
    axes[0, 1].axhline(y=0, color="green", linestyle="--", alpha=0.5, linewidth=1.5)
    axes[0, 1].axvline(x=0, color="gray", linestyle="--", alpha=0.4)

    # Highlight zero-crossings
    if analysis["zero_crossings"]:
        zc_alphas = [r.alpha for r in analysis["zero_crossings"]]
        zc_returns = [r.functional_return for r in analysis["zero_crossings"]]
        axes[0, 1].scatter(
            zc_alphas,
            zc_returns,
            color="green",
            s=150,
            zorder=5,
            marker="*",
            edgecolors="black",
            linewidth=1,
        )

    axes[0, 1].set_xlabel("α", fontsize=12)
    axes[0, 1].set_ylabel("|L(α) - L(M_base)|", fontsize=12)
    axes[0, 1].set_title("Functional Return", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)

    # ─── Plot 3: Signed Delta L(α) - L_base ───
    deltas = [r.loss - r.base_loss for r in results]
    axes[1, 0].plot(alphas, deltas, "b-", linewidth=2, alpha=0.8)
    axes[1, 0].axhline(y=0, color="red", linestyle="--", linewidth=2, alpha=0.7)
    axes[1, 0].axvline(x=0, color="gray", linestyle="--", alpha=0.4)
    axes[1, 0].fill_between(alphas, 0, deltas, alpha=0.2, color="blue")

    # Highlight zero-crossings
    if analysis["zero_crossings"]:
        zc_alphas = [r.alpha for r in analysis["zero_crossings"]]
        zc_deltas = [r.loss - r.base_loss for r in analysis["zero_crossings"]]
        axes[1, 0].scatter(
            zc_alphas,
            zc_deltas,
            color="green",
            s=150,
            zorder=5,
            marker="*",
            edgecolors="black",
            linewidth=1,
        )

    axes[1, 0].set_xlabel("α", fontsize=12)
    axes[1, 0].set_ylabel("L(α) - L(M_base)", fontsize=12)
    axes[1, 0].set_title("Signed Loss Delta", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    # ─── Plot 4: Task Performance vs α ───
    axes[1, 1].plot(alphas, task_perfs, "g-", linewidth=2, alpha=0.8, label="Task Loss")
    axes[1, 1].axhline(
        y=base_loss, color="gray", linestyle="--", alpha=0.5, label="Base Loss"
    )
    axes[1, 1].axvline(x=0, color="gray", linestyle="--", alpha=0.4)

    axes[1, 1].set_xlabel("α", fontsize=12)
    axes[1, 1].set_ylabel("Task-Specific Loss", fontsize=12)
    axes[1, 1].set_title("Task Performance", fontsize=12, fontweight="bold")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    # NEW: Squaring test plots (Enhancement #3)
    if enable_squaring_test:
        # ─── Plot 5: Squaring Test - L(α) vs L(2α) ───
        axes[0, 2].plot(alphas, losses, "b-", linewidth=2.5, label="L(α)", alpha=0.8)
        axes[0, 2].plot(alphas, losses_2alpha, "r-", linewidth=2.5, label="L(2α)", alpha=0.8)
        axes[0, 2].axhline(
            y=base_loss,
            color="green",
            linestyle="--",
            linewidth=2,
            label="L(M_base)",
            alpha=0.7,
        )
        axes[0, 2].axvline(x=0, color="gray", linestyle="--", alpha=0.4)

        # Find squaring return points (where L(2α) ≈ L_base)
        if "squaring_return_points" in analysis:
            sr_alphas = [r.alpha for r in analysis["squaring_return_points"]]
            sr_losses_2alpha = [r.loss_2alpha for r in analysis["squaring_return_points"]]
            axes[0, 2].scatter(
                sr_alphas,
                sr_losses_2alpha,
                color="orange",
                s=150,
                zorder=5,
                marker="*",
                edgecolors="black",
                linewidth=1,
                label="Squaring returns",
            )

        axes[0, 2].set_xlabel("α", fontsize=12)
        axes[0, 2].set_ylabel("Loss", fontsize=12)
        axes[0, 2].set_title("Squaring Test: [W(λ)]² = I Analog", fontsize=12, fontweight="bold")
        axes[0, 2].legend(fontsize=8, loc="best")
        axes[0, 2].grid(True, alpha=0.3)

        # ─── Plot 6: Squaring Functional Return |L(2α) - L_base| ───
        axes[1, 2].plot(alphas, functional_returns_2alpha, "r-", linewidth=2.5, alpha=0.8, label="|L(2α) - L_base|")
        axes[1, 2].plot(alphas, functional_returns, "b--", linewidth=1.5, alpha=0.5, label="|L(α) - L_base|")
        axes[1, 2].axhline(y=0, color="green", linestyle="--", alpha=0.5, linewidth=1.5)
        axes[1, 2].axvline(x=0, color="gray", linestyle="--", alpha=0.4)

        # Highlight squaring return points
        if "squaring_return_points" in analysis:
            sr_alphas = [r.alpha for r in analysis["squaring_return_points"]]
            sr_returns_2alpha = [r.functional_return_2alpha for r in analysis["squaring_return_points"]]
            axes[1, 2].scatter(
                sr_alphas,
                sr_returns_2alpha,
                color="orange",
                s=150,
                zorder=5,
                marker="*",
                edgecolors="black",
                linewidth=1,
            )

        axes[1, 2].set_xlabel("α", fontsize=12)
        axes[1, 2].set_ylabel("|L(2α) - L(M_base)|", fontsize=12)
        axes[1, 2].set_title("Squaring Functional Return", fontsize=12, fontweight="bold")
        axes[1, 2].legend(fontsize=8, loc="best")
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Plot saved to {output_path}")
    plt.close()


def plot_2d_composition(
    results: list[TwoDSweepResult],
    task1_name: str,
    task2_name: str,
    output_path: str = "loss_landscape_2d.png",
):
    """Visualize 2D composition M(α,β) = M_base + α·T1 + β·T2 (Enhancement #2).

    Creates a heatmap showing the loss landscape in (α, β) space.
    Tests for rotation-like structure under task vector composition.
    """
    # Extract data
    alphas = sorted(list(set(r.alpha for r in results)))
    betas = sorted(list(set(r.beta for r in results)))

    # Create 2D grids
    alpha_grid, beta_grid = np.meshgrid(alphas, betas)
    loss_grid = np.zeros_like(alpha_grid)
    functional_return_grid = np.zeros_like(alpha_grid)

    # Fill grids
    for r in results:
        i = alphas.index(r.alpha)
        j = betas.index(r.beta)
        loss_grid[j, i] = r.loss
        functional_return_grid[j, i] = r.functional_return

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        f"2D Task Vector Composition: M(α,β) = M_base + α·T1 + β·T2\n"
        f"T1: {task1_name}, T2: {task2_name}",
        fontsize=14,
        fontweight="bold",
    )

    # Plot 1: Loss heatmap
    im1 = axes[0].contourf(alpha_grid, beta_grid, loss_grid, levels=20, cmap="viridis")
    axes[0].contour(alpha_grid, beta_grid, loss_grid, levels=10, colors="white", alpha=0.3, linewidths=0.5)
    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    axes[0].scatter([0], [0], color="red", s=100, marker="X", edgecolors="white", linewidth=2, label="M_base", zorder=10)
    axes[0].set_xlabel(f"α (scaling for {task1_name})", fontsize=12)
    axes[0].set_ylabel(f"β (scaling for {task2_name})", fontsize=12)
    axes[0].set_title("Loss Landscape L(α,β)", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)
    fig.colorbar(im1, ax=axes[0], label="Loss")

    # Plot 2: Functional return heatmap
    im2 = axes[1].contourf(alpha_grid, beta_grid, functional_return_grid, levels=20, cmap="RdYlGn_r")
    axes[1].contour(alpha_grid, beta_grid, functional_return_grid, levels=10, colors="white", alpha=0.3, linewidths=0.5)
    axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.5, linewidth=1)
    axes[1].scatter([0], [0], color="red", s=100, marker="X", edgecolors="white", linewidth=2, label="M_base", zorder=10)
    axes[1].set_xlabel(f"α (scaling for {task1_name})", fontsize=12)
    axes[1].set_ylabel(f"β (scaling for {task2_name})", fontsize=12)
    axes[1].set_title("Functional Return |L(α,β) - L_base|", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)
    fig.colorbar(im2, ax=axes[1], label="|ΔL|")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 2D composition plot saved to {output_path}")
    plt.close()


def main():
    """Sweep loss landscape along task vector direction.

    Directly visualizes L(M_base + αT) for α ∈ [-3, 3].
    Answers: Does the loss curve cross L(M_base) at any α ≠ 0?

    Inspired by Eckmann & Tlusty (2025)'s rotation group self-inverse walks.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Task Vector Loss Landscape Explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment (fine-tuning + analysis)
  python main.py

  # Run analysis only on previously fine-tuned models
  python main.py --analysis-only

  # Run with custom alpha range and samples
  python main.py --alpha-min -5 --alpha-max 5 --num-samples 200

  # Analysis only with custom parameters
  python main.py --analysis-only --alpha-min -2 --alpha-max 2 --num-samples 50
        """,
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip fine-tuning and run analysis on previously saved models",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=-3.0,
        help="Minimum alpha value for sweep (default: -3.0)",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        default=3.0,
        help="Maximum alpha value for sweep (default: 3.0)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of alpha samples (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory (default: ./outputs)",
    )
    # NEW: Enhancement flags
    parser.add_argument(
        "--enable-squaring-test",
        action="store_true",
        default=True,
        help="Enable squaring test: evaluate M(2α) to test [W(λ)]² = I analog (default: enabled)",
    )
    parser.add_argument(
        "--disable-squaring-test",
        action="store_true",
        help="Disable squaring test to speed up sweep",
    )
    parser.add_argument(
        "--enable-2d-composition",
        action="store_true",
        help="Enable 2D task vector composition experiment M(α,β) = M_base + α·T1 + β·T2",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sentiment_positive",
        choices=["sentiment_positive", "sentiment_negative", "instruction_following", "qa_factual"],
        help="Task to use for fine-tuning (default: sentiment_positive)",
    )
    parser.add_argument(
        "--multi-task",
        action="store_true",
        help="Run experiment on all available tasks for comparison",
    )
    parser.add_argument(
        "--2d-samples",
        type=int,
        default=20,
        help="Samples per dimension for 2D composition (creates NxN grid, default: 20)",
    )

    args = parser.parse_args()

    # Handle squaring test flag logic
    enable_squaring_test = args.enable_squaring_test and not args.disable_squaring_test

    experiment_start_time = time.time()

    print(f"\n{'='*70}")
    print("TASK VECTOR LOSS LANDSCAPE EXPERIMENT")
    print("Inspired by Eckmann & Tlusty (2025)")
    print(f"{'='*70}")

    if args.analysis_only:
        print("\n🔍 MODE: Analysis Only (using saved models)")
    else:
        print("\n🚀 MODE: Full Experiment (fine-tuning + analysis)")

    print(
        """
RESEARCH QUESTION:
─────────────────
What does the loss look like along the 1D line M(α) = M_base + αT?

Specifically: Does the loss curve cross L(M_base) at any α ≠ 0?

THE EXPERIMENT:
──────────────
1. Create task vector: T = M_finetuned - M_base
2. Sweep α from -3.0 to 3.0 (100 samples)
3. For each α: evaluate L(M_base + αT)
4. Plot L(α) vs α

WHAT WE'RE LOOKING FOR:
──────────────────────
• Zero-crossings: α ≠ 0 where L(α) ≈ L(M_base) ("functional return")
• Minimum loss: Best α for task performance
• Loss landscape shape: Monotonic? Periodic? Symmetric?

CONNECTION TO PAPER:
───────────────────
Eckmann & Tlusty prove that rotation group walks [W(λ)]² = I have abundant
special λ values (180° rotations). We test if neural loss landscapes exhibit
analogous structure under task vector scaling.
"""
    )

    # Configuration
    model_name = "google/gemma-3-4b-it"
    output_dir = args.output_dir
    alpha_range = (args.alpha_min, args.alpha_max)
    num_samples = args.num_samples

    # Device detection with MPS support for Apple Silicon
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    os.makedirs(output_dir, exist_ok=True)

    # Initialize experiment metrics
    experiment_metrics = ExperimentMetrics(
        start_time=datetime.fromtimestamp(experiment_start_time).isoformat(),
        model_name=model_name,
        device=device,
    )

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Output dir: {output_dir}")
    print(f"  Alpha range: [{alpha_range[0]:.1f}, {alpha_range[1]:.1f}]")
    print(f"  Num samples: {num_samples}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare task data (sentiment analysis)
    print("\nPreparing task data (sentiment analysis)...")

    # Training data - positive sentiment examples
    train_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "The product exceeded all my expectations. Highly recommended!",
        "What a wonderful experience! I'm so happy with this purchase.",
        "Amazing quality and great value. I'm very satisfied.",
        "This is the best thing I've bought in years. Incredible!",
        "Outstanding service and excellent results. Very pleased!",
        "I'm delighted with this purchase. It works perfectly.",
        "Exceptional quality and superb performance. Love it!",
        "This is exactly what I needed. Fantastic product!",
        "Brilliant! This has made my life so much easier.",
    ] * 3  # Repeat for more training data

    # Task evaluation texts (positive sentiment)
    task_eval_texts = [
        "This product is amazing and works great!",
        "I'm very happy with my purchase. Excellent quality!",
        "Wonderful experience! Highly recommended.",
        "Best purchase I've made this year!",
        "Fantastic quality and great value.",
    ]

    # General evaluation texts (neutral/mixed)
    general_eval_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "The weather today is quite different from yesterday.",
        "Programming requires logical thinking and problem solving.",
        "The capital of France is Paris.",
    ]

    # Check if we should load existing models or fine-tune new ones
    if args.analysis_only:
        if check_saved_models_exist(output_dir):
            print(f"\n{'='*70}")
            print("LOADING SAVED MODELS")
            print(f"{'='*70}")
            base_model, finetuned_model = load_models_for_analysis(
                model_name, output_dir, device
            )
            # Set dummy metrics for analysis-only mode
            experiment_metrics.training_examples = len(train_texts)
            experiment_metrics.num_epochs = 2  # Default from previous run
            experiment_metrics.learning_rate = 5e-5  # Default from previous run
        else:
            print(f"\n{'='*70}")
            print("WARNING: No saved models found!")
            print(f"{'='*70}")
            print(f"Looking for models in: {output_dir}")
            print("  - saved_base_model/")
            print("  - saved_finetuned_model/")
            print("\nPlease run the full experiment first without --analysis-only")
            print("Or check that the --output-dir path is correct.")
            return
    else:
        # Full experiment mode: Load base model and fine-tune
        print(f"\n{'='*70}")
        print("LOADING BASE MODEL")
        print(f"{'='*70}")
        device_map = get_device_map()
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device_map
        )

        # For MPS/CPU, manually move to device if device_map is None
        if device_map is None and device != "cpu":
            base_model = base_model.to(device)  # type: ignore[assignment]

        # Fine-tune model on task
        # NOTE: We don't use device_map="auto" for fine-tuning to avoid meta device issues
        print("\nLoading model for fine-tuning...")
        ft_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=None
        )
        ft_model = ft_model.to(device)  # type: ignore[assignment]

        finetuned_model, ft_metrics = fine_tune_model(
            base_model=ft_model,
            tokenizer=tokenizer,
            train_texts=train_texts,
            output_dir=f"{output_dir}/finetuned_tmp",
            num_epochs=2,
            learning_rate=5e-5,
        )

        # Update experiment metrics with fine-tuning data
        experiment_metrics.finetuning_start_time = ft_metrics["start_time"]
        experiment_metrics.finetuning_end_time = ft_metrics["end_time"]
        experiment_metrics.finetuning_duration_seconds = ft_metrics["duration_seconds"]
        experiment_metrics.training_examples = ft_metrics["training_examples"]
        experiment_metrics.num_epochs = ft_metrics["num_epochs"]
        experiment_metrics.learning_rate = ft_metrics["learning_rate"]
        experiment_metrics.final_training_loss = ft_metrics["final_loss"]
        experiment_metrics.training_steps = ft_metrics["training_steps"]
        experiment_metrics.training_history = ft_metrics["training_history"]

        # Save models for future analysis
        save_models_for_analysis(base_model, finetuned_model, output_dir)

    # Compute task vector
    print("Computing task vector T = M_finetuned - M_base...")
    tv_start = time.time()
    task_vector = compute_task_vector(base_model, finetuned_model)
    tv_elapsed = time.time() - tv_start

    # Calculate task vector magnitude
    tv_magnitude = np.sqrt(sum((v**2).sum().item() for v in task_vector.values()))
    print(f"Task vector magnitude: ||T|| = {tv_magnitude:.4f}")
    print(f"Computation time: {tv_elapsed:.2f}s\n")

    # Update experiment metrics
    experiment_metrics.task_vector_magnitude = tv_magnitude
    experiment_metrics.task_vector_computation_time = tv_elapsed

    # Count model parameters
    num_params = sum(p.numel() for p in base_model.parameters())
    experiment_metrics.model_parameters = num_params
    print(f"Model parameters: {num_params:,}\n")

    # Run experiment
    results, sweep_metrics = sweep_alpha_values(
        base_model=base_model,
        task_vector=task_vector,
        tokenizer=tokenizer,
        general_eval_texts=general_eval_texts,
        task_eval_texts=task_eval_texts,
        alpha_range=alpha_range,
        num_samples=num_samples,
        device=device,
        enable_squaring_test=enable_squaring_test,  # NEW
    )

    # Update experiment metrics with sweep timing
    experiment_metrics.sweep_start_time = sweep_metrics["start_time"]
    experiment_metrics.sweep_end_time = sweep_metrics["end_time"]
    experiment_metrics.sweep_duration_seconds = sweep_metrics["duration_seconds"]
    experiment_metrics.time_per_alpha_seconds = sweep_metrics["time_per_alpha_seconds"]

    # Analyze results
    analysis = analyze_results(results)

    # Update experiment metrics with sweep and results data
    min_general = analysis["min_general_loss"]
    min_task = analysis["min_task_loss"]
    zero_crossings = analysis["zero_crossings"]

    experiment_metrics.num_alpha_samples = len(results)
    experiment_metrics.alpha_range = alpha_range
    experiment_metrics.min_general_loss_alpha = min_general.alpha
    experiment_metrics.min_general_loss = min_general.loss
    experiment_metrics.min_task_loss_alpha = min_task.alpha
    experiment_metrics.min_task_loss = min_task.task_performance
    experiment_metrics.num_zero_crossings = len(zero_crossings)
    experiment_metrics.zero_crossing_alphas = [zc.alpha for zc in zero_crossings]

    # Generate markdown report
    generate_markdown_report(
        experiment_metrics=experiment_metrics,
        analysis=analysis,
        results=results,
        output_path=f"{output_dir}/experiment_report.md",
    )
    print(f"📝 Markdown report saved to {output_dir}/experiment_report.md")

    # Plot results
    plot_results(results, analysis, f"{output_dir}/loss_landscape_sweep.png", enable_squaring_test=enable_squaring_test)

    # Save detailed results
    results_path = f"{output_dir}/loss_landscape_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "paper_reference": "Eckmann & Tlusty (2025), arXiv:2502.14367v3",
                    "description": "Loss landscape sweep along task vector direction",
                    "task": "sentiment_analysis",
                    "task_vector_magnitude": tv_magnitude,
                    "model": model_name,
                    "alpha_range": list(alpha_range),
                    "num_samples": num_samples,
                },
                "methodology": {
                    "experiment": "M(α) = M_base + αT",
                    "primary_metric": "L(α) = loss of M(α)",
                    "question": "Does L(α) cross L(M_base) at any α ≠ 0?",
                },
                "key_results": {
                    "min_general_loss": asdict(analysis["min_general_loss"]),
                    "min_task_loss": asdict(analysis["min_task_loss"]),
                    "best_return": asdict(analysis["best_return"]),
                    "zero_crossings": [asdict(r) for r in analysis["zero_crossings"]],
                },
                "top_10_by_return": [asdict(r) for r in analysis["sorted_by_return"]],
                "all_results": [asdict(r) for r in results],
            },
            f,
            indent=2,
        )
    print(f"📄 Results saved to {results_path}")

    # Complete experiment metrics and save
    experiment_end_time = time.time()
    experiment_metrics.end_time = datetime.fromtimestamp(experiment_end_time).isoformat()
    experiment_metrics.duration_seconds = experiment_end_time - experiment_start_time

    metrics_path = f"{output_dir}/experiment_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(asdict(experiment_metrics), f, indent=2)
    print(f"📊 Experiment metrics saved to {metrics_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    min_general = analysis["min_general_loss"]
    min_task = analysis["min_task_loss"]
    best_return = analysis["best_return"]
    zero_crossings = analysis["zero_crossings"]

    print("MINIMUM GENERAL LOSS (Best General Knowledge):")
    print(f"  α = {min_general.alpha:+.4f}")
    print(f"  L(α) = {min_general.loss:.4f}")
    print(f"  L(M_base) = {min_general.base_loss:.4f}")
    print(f"  Improvement: {min_general.base_loss - min_general.loss:+.4f}")

    print("\nMINIMUM TASK LOSS (Best Task Performance):")
    print(f"  α = {min_task.alpha:+.4f}")
    print(f"  Task L(α) = {min_task.task_performance:.4f}")
    print(f"  General L(α) = {min_task.loss:.4f}")

    if abs(min_general.alpha - min_task.alpha) > 0.1:
        print("\n  ⚠️  TRADE-OFF DETECTED!")
        print(f"  → Best general knowledge at α = {min_general.alpha:+.4f}")
        print(f"  → Best task performance at α = {min_task.alpha:+.4f}")
        print(f"  → Δα = {abs(min_task.alpha - min_general.alpha):.4f}")
    else:
        print("\n  ✓ Both minimums occur at similar α values")

    print("\nBEST FUNCTIONAL RETURN (smallest |L(α) - L_base|):")
    print(f"  α = {best_return.alpha:+.4f}")
    print(f"  |ΔL| = {best_return.functional_return:.6f}")
    print(f"  L(α) = {best_return.loss:.4f}")

    print("\nZERO-CROSSINGS (α ≠ 0 where L(α) ≈ L_base):")
    if zero_crossings:
        print(f"  ✓ Found {len(zero_crossings)} crossing(s)!")
        for i, zc in enumerate(zero_crossings[:5], 1):
            print(
                f"     {i}. α = {zc.alpha:+.4f}, L(α) = {zc.loss:.4f}, |ΔL| = {zc.functional_return:.6f}"
            )
    else:
        print("  ✗ No zero-crossings found")
        print("  → Loss is monotonic along task vector direction")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print(
        """
CONNECTION TO PAPER:
───────────────────
Eckmann & Tlusty (2025) prove that rotation group walks have abundant special
λ values where [W(λ)]² = I (self-inverse property). This happens when W(λ) is
a 180° rotation: R(n,π)² = I.

OUR EXPERIMENT:
──────────────
We tested if neural loss landscapes exhibit analogous structure by sweeping
L(M_base + αT) along the task vector direction.

Question: Does L(α) cross L(M_base) at any α ≠ 0?

If YES → suggests special scaling factors exist (analogous to 180° rotations)
If NO  → loss is monotonic (task vectors lack rotation-like symmetry)

KEY DIFFERENCES FROM PAPER:
───────────────────────────
⚠️  Rotations: Multiplicative group (W₁ ∘ W₂)
⚠️  Task vectors: Additive vector space (v₁ + v₂)
⚠️  No group structure → no proven abundance of special α values
⚠️  This is EMPIRICAL exploration, not a theorem

PRACTICAL IMPLICATIONS:
──────────────────────
• Minimum α: Best scaling for task performance
• Zero-crossings: Special α values for model merging
• Landscape shape: Insights into parameter space geometry
"""
    )

    # Final experiment metrics summary
    total_duration = experiment_metrics.duration_seconds
    print(f"\n{'='*70}")
    print("EXPERIMENT METRICS SUMMARY")
    print(f"{'='*70}\n")

    print(f"Total Duration: {total_duration / 60:.1f} minutes ({total_duration:.0f}s)")
    print(f"  - Fine-tuning: {experiment_metrics.finetuning_duration_seconds / 60:.1f}m "
          f"({experiment_metrics.finetuning_duration_seconds / total_duration * 100:.1f}%)")
    print(f"  - Alpha sweep: {experiment_metrics.sweep_duration_seconds / 60:.1f}m "
          f"({experiment_metrics.sweep_duration_seconds / total_duration * 100:.1f}%)")

    print(f"\nModel: {experiment_metrics.model_name}")
    print(f"  Parameters: {experiment_metrics.model_parameters:,}")
    print(f"  Device: {experiment_metrics.device}")

    print(f"\nFine-tuning:")
    print(f"  Training examples: {experiment_metrics.training_examples}")
    print(f"  Epochs: {experiment_metrics.num_epochs}")
    print(f"  Steps: {experiment_metrics.training_steps}")
    print(f"  Learning rate: {experiment_metrics.learning_rate:.2e}")
    print(f"  Final loss: {experiment_metrics.final_training_loss:.4f}")
    if experiment_metrics.training_steps > 0:
        print(f"  Avg time/step: {experiment_metrics.finetuning_duration_seconds / experiment_metrics.training_steps:.2f}s")
    else:
        print(f"  Avg time/step: N/A (analysis-only mode)")

    print(f"\nTask Vector:")
    print(f"  Magnitude ||T||: {experiment_metrics.task_vector_magnitude:.4f}")
    print(f"  Computation time: {experiment_metrics.task_vector_computation_time:.2f}s")

    print(f"\nAlpha Sweep:")
    print(f"  Samples: {experiment_metrics.num_alpha_samples}")
    print(f"  Range: [{experiment_metrics.alpha_range[0]:.1f}, {experiment_metrics.alpha_range[1]:.1f}]")
    print(f"  Avg time/sample: {experiment_metrics.time_per_alpha_seconds:.2f}s")

    print(f"\nKey Results:")
    print(f"  Min general loss α: {experiment_metrics.min_general_loss_alpha:+.4f} (L={experiment_metrics.min_general_loss:.4f})")
    print(f"  Min task loss α: {experiment_metrics.min_task_loss_alpha:+.4f} (L={experiment_metrics.min_task_loss:.4f})")
    print(f"  Zero-crossings: {experiment_metrics.num_zero_crossings}")
    if experiment_metrics.zero_crossing_alphas:
        print(f"  Zero-crossing α values: {', '.join(f'{a:+.4f}' for a in experiment_metrics.zero_crossing_alphas[:5])}")

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Started:  {experiment_metrics.start_time}")
    print(f"Finished: {experiment_metrics.end_time}")
    print(f"Duration: {total_duration / 60:.1f} minutes")
    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  - {output_dir}/experiment_report.md (comprehensive analysis)")
    print(f"  - {output_dir}/loss_landscape_sweep.png (visualization)")
    print(f"  - {output_dir}/loss_landscape_results.json (detailed data)")
    print(f"  - {output_dir}/experiment_metrics.json (metrics)")
    if not args.analysis_only:
        print(f"  - {output_dir}/saved_base_model/ (saved for re-analysis)")
        print(f"  - {output_dir}/saved_finetuned_model/ (saved for re-analysis)")
    print(f"{'='*70}\n")
    print("📋 Copy experiment_report.md to share with LLMs for analysis!")
    if not args.analysis_only:
        print("\n💡 TIP: To re-run analysis with different parameters without re-training:")
        print(f"   python main.py --analysis-only --alpha-min -5 --alpha-max 5 --num-samples 200")


if __name__ == "__main__":
    main()
