"""
Self-Inverse Task Vectors: Exploring Functional Return in Neural Network Parameter Space
Inspired by "Walks in Rotation Spaces Return Home when Doubled and Scaled"
(Eckmann & Tlusty, 2025, arXiv:2502.14367v3)

MATHEMATICAL BACKGROUND:
─────────────────────────
Eckmann & Tlusty prove that for rotation groups SO(3)/SU(2), almost any walk W
can be scaled and doubled to return to identity: [W(λ)]² = I for special λ values.

Key insight: Random rotations by small angles are rare (f₁(ω) ∝ 1-cosω → 0 as ω→0),
but 180° rotations are abundant (f₁(π) = 2/π). When a walk W(λ) reaches a 180°
rotation, squaring it returns to identity: R(n,π)² = I.

CRITICAL STRUCTURAL DIFFERENCES:
─────────────────────────────────
⚠️  Rotations form a MULTIPLICATIVE GROUP:
    • Operation: W₁ ∘ W₂ (composition)
    • Identity: I (identity rotation)
    • Doubling: W² = W ∘ W (apply rotation, then apply it again)

⚠️  Task vectors form an ADDITIVE VECTOR SPACE:
    • Operation: v₁ + v₂ (vector addition)
    • Identity: 0 (zero vector, or M_base in model space)
    • "Doubling": Unclear! Two interpretations:
        (a) 2λT (scalar multiplication) - NOT analogous to W²
        (b) Compositional: Apply λT, extract induced transformation, apply again

RESEARCH QUESTION:
──────────────────
We test whether neural network loss landscapes exhibit analogous "functional roots of identity" under task vector transformations:

PRIMARY EXPERIMENT (Compositional - closer to paper):
  M₁ = M_base + λT                    (first application)
  T₁ = M₁ - M_base = λT               (induced change)
  M₂ = M₁ + λT₁ = M_base + λT + λ²T  (second application)

  Question: ∃ λ ≠ 0 such that L(M₂) ≈ L(M_base)?

  This tests if loss landscape has special "fixpoint-like" λ values where
  iterative application returns functionally (not geometrically) to base.

SECONDARY EXPERIMENT (Linear - simpler but less analogous):
  M_doubled = M_base + 2λT

  Question: ∃ λ ≠ 0 such that L(M_doubled) ≈ L(M_base)?

  This tests if loss landscape is periodic/symmetric along task vector direction.
  Note: Geometric distance ||M_doubled - M_base|| = 2|λ|·||T|| is trivially
  minimized at λ=0, so we measure functional return: |L(M₂) - L(M_base)|.

RELATIONSHIP TO PAPER:
──────────────────────
• Direct analogy: COMPOSITIONAL version (iterative application)
• Weaker connection: LINEAR version (scalar scaling)
• Both explore "functional return" in loss landscape
• Key difference: We lack the group structure and Haar measure that make the
  paper's result provable. This is an empirical exploration.

POTENTIAL FINDINGS:
───────────────────
If non-zero λ* exist where functional return ≈ 0:
  ✓ Indicates rich geometric structure in loss landscapes
  ✓ Suggests special scaling factors for task vector arithmetic
  ✓ Applications in model merging, multi-task learning
  ✓ Hints at deeper algebraic structure in parameter space

If no such λ* exist:
  ✓ Indicates loss landscape is monotonic along task vector direction
  ✓ Suggests task vectors lack the symmetry properties of rotations
  ✓ Still provides insights into parameter space geometry
"""

import json
import os
from dataclasses import asdict, dataclass
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
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return None  # MPS: load to default device, then move manually
    else:
        return None  # CPU: load to default device


@dataclass
class InverseResult:
    """Store comprehensive results for a specific λ value.

    Tests both linear (2λT) and compositional (λT + λ²T) versions.
    """
    lambda_val: float

    # ─── GEOMETRIC METRICS (for reference) ───
    geometric_distance_linear: float       # ||M_base + 2λT - M_base|| = 2|λ|·||T||
    geometric_distance_compositional: float  # ||M_base + λT + λ²T - M_base||

    # ─── FUNCTIONAL METRICS (PRIMARY) ───
    base_loss: float                       # L(M_base) - reference point
    intermediate_loss: float               # L(M_base + λT) - first application

    # Linear version: M_doubled = M_base + 2λT
    doubled_loss_linear: float             # L(M_base + 2λT)
    functional_return_linear: float        # |L(M_base + 2λT) - L(M_base)|

    # Compositional version: M_comp = M_base + λT + λ²T (closer to paper)
    doubled_loss_compositional: float      # L(M_base + λT + λ²T)
    functional_return_compositional: float  # |L(M_base + λT + λ²T) - L(M_base)|

    # ─── UTILITY METRICS ───
    intermediate_task_performance: float   # Task loss at M_base + λT
    utility_score_linear: float            # Combined score for linear version
    utility_score_compositional: float     # Combined score for compositional version

    # ─── STABILITY METRICS ───
    loss_curvature: float | None = None  # d²L/dλ² estimate


def compute_model_distance(model1: nn.Module, model2: nn.Module) -> float:
    """Compute normalized L2 distance between model parameters.

    Returns:
        float: RMS distance per parameter
    """
    distance = 0.0
    param_count = 0

    for p1, p2 in zip(model1.parameters(), model2.parameters(), strict=True):
        distance += torch.sum((p1 - p2) ** 2).item()
        param_count += p1.numel()

    return float(np.sqrt(distance / param_count))


def apply_task_vector(
    base_model: PreTrainedModel,
    task_vector: dict[str, torch.Tensor],
    lambda_scale: float,
    device: str = "cpu"
) -> PreTrainedModel:
    """Apply scaled task vector: M_new = M_base + λT.

    Args:
        base_model: Base model (defines M_base)
        task_vector: Task vector T = M_finetuned - M_base
        lambda_scale: Scaling factor λ
        device: Device to place model on

    Returns:
        New model with parameters M_base + λT
    """
    model_name = getattr(base_model.config, 'name_or_path', 'gpt2')
    device_map = get_device_map()
    model = type(base_model).from_pretrained(  # type: ignore[attr-defined]
        model_name,
        torch_dtype=base_model.dtype,
        device_map=device_map
    )

    # For MPS/CPU, manually move to device if device_map is None
    if device_map is None and device != "cpu":
        model = model.to(device)  # type: ignore[assignment]

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in task_vector:
                param.copy_(param + lambda_scale * task_vector[name].to(param.device))

    return model  # type: ignore[return-value]


def apply_task_vector_compositional(
    base_model: PreTrainedModel,
    task_vector: dict[str, torch.Tensor],
    lambda_scale: float,
    device: str = "cpu"
) -> PreTrainedModel:
    """Apply task vector compositionally: M_comp = M_base + λT + λ²T.

    This is closer to the paper's notion of W² = W ∘ W (composition).
    We interpret "applying the transformation twice" as:
        1. M₁ = M_base + λT        (first application)
        2. T₁ = M₁ - M_base = λT   (induced transformation)
        3. M₂ = M₁ + λ·T₁ = M_base + λT + λ²T  (second application)

    Args:
        base_model: Base model (defines M_base)
        task_vector: Task vector T = M_finetuned - M_base
        lambda_scale: Scaling factor λ
        device: Device to place model on

    Returns:
        New model with parameters M_base + λT + λ²T
    """
    model_name = getattr(base_model.config, 'name_or_path', 'gpt2')
    device_map = get_device_map()
    model = type(base_model).from_pretrained(  # type: ignore[attr-defined]
        model_name,
        torch_dtype=base_model.dtype,
        device_map=device_map
    )

    # For MPS/CPU, manually move to device if device_map is None
    if device_map is None and device != "cpu":
        model = model.to(device)  # type: ignore[assignment]

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in task_vector:
                # M_comp = M_base + λT + λ²T = M_base + λT(1 + λ)
                param.copy_(param + lambda_scale * (1 + lambda_scale) * task_vector[name].to(param.device))

    return model  # type: ignore[return-value]


def compute_task_vector(
    base_model: PreTrainedModel,
    finetuned_model: PreTrainedModel
) -> dict[str, torch.Tensor]:
    """Compute task vector: T = M_finetuned - M_base"""
    task_vector = {}

    for (name, base_param), (_, ft_param) in zip(
        base_model.named_parameters(),
        finetuned_model.named_parameters(),
        strict=True
    ):
        task_vector[name] = (ft_param - base_param).clone().cpu()

    return task_vector


def evaluate_model(
    model: nn.Module,
    tokenizer,
    eval_texts: list[str],
    device: str = "cuda"
) -> float:
    """Evaluate model perplexity on evaluation texts"""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            count += 1

    return total_loss / count


def evaluate_task_performance(
    model: nn.Module,
    tokenizer,
    task_eval_texts: list[str],
    device: str = "cuda"
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
    learning_rate: float = 5e-5
) -> PreTrainedModel:
    """Fine-tune model on task to create a structured task vector.

    Args:
        base_model: Pre-trained base model
        tokenizer: Associated tokenizer
        train_texts: Training examples for the task
        output_dir: Directory for checkpoints
        num_epochs: Number of training epochs
        learning_rate: Learning rate for fine-tuning

    Returns:
        Fine-tuned model M_finetuned, where T = M_finetuned - M_base
    """
    print("\nFine-tuning model on task data...")
    print(f"  Training examples: {len(train_texts)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")

    # Prepare dataset
    class TextDataset(Dataset):
        """Dataset wrapper for tokenizing text inputs for language model training."""

        def __init__(self, texts, tokenizer, max_length=512):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )

        def __len__(self):
            return len(self.encodings.input_ids)

        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings.input_ids[idx],
                'attention_mask': self.encodings.attention_mask[idx]
            }

    dataset = TextDataset(train_texts, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,  # Reduced from 4 for 12B model
        learning_rate=learning_rate,
        save_steps=10000,  # Don't save intermediate checkpoints
        save_total_limit=1,
        logging_steps=50,
        report_to=[],  # Disable wandb/tensorboard
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Save memory for large model
        fp16=False,  # Using bfloat16 instead
        bf16=True,  # Match model's native precision
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Trainer
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    print("Fine-tuning complete!\n")
    return cast(PreTrainedModel, trainer.model)


def find_inverse_lambda_values(
    base_model: PreTrainedModel,
    task_vector: dict[str, torch.Tensor],
    tokenizer,
    general_eval_texts: list[str],
    task_eval_texts: list[str],
    lambda_range: tuple[float, float] = (-2.0, 2.0),
    num_samples: int = 50,
    device: str = "cuda"
) -> list[InverseResult]:
    """Search for λ values where functional return ≈ 0.

    Tests both approaches:
    1. LINEAR: M_doubled = M_base + 2λT (simpler, but less analogous)
    2. COMPOSITIONAL: M_comp = M_base + λT + λ²T (closer to paper's W²)

    Args:
        base_model: Base model M_base
        task_vector: Task vector T = M_finetuned - M_base
        tokenizer: Tokenizer for evaluation
        general_eval_texts: Neutral evaluation texts
        task_eval_texts: Task-specific evaluation texts
        lambda_range: Range of λ values to test
        num_samples: Number of λ samples
        device: Device for computation

    Returns:
        List of InverseResult objects with metrics for each λ
    """

    print(f"\n{'='*70}")
    print("SEARCHING FOR FUNCTIONAL SELF-INVERSE λ VALUES")
    print(f"{'='*70}")
    print(f"Range: {lambda_range}")
    print(f"Samples: {num_samples}")
    print("\nTesting both LINEAR and COMPOSITIONAL versions:")
    print("  • LINEAR:        L(M_base + 2λT) ≈? L(M_base)")
    print("  • COMPOSITIONAL: L(M_base + λT + λ²T) ≈? L(M_base)")
    print("\nSearching for non-zero λ with small functional return...\n")

    lambda_values = np.linspace(lambda_range[0], lambda_range[1], num_samples)
    results = []

    # Compute base model loss for reference
    base_model.to(device)
    base_loss = evaluate_model(base_model, tokenizer, general_eval_texts, device)
    print(f"Base model loss (general eval): {base_loss:.4f}\n")

    for i, lambda_val in enumerate(lambda_values):
        print(f"[{i+1}/{num_samples}] λ = {lambda_val:+.3f} ", end="", flush=True)

        # ─── Create models ───
        intermediate_model = apply_task_vector(base_model, task_vector, lambda_val, device)
        doubled_linear = apply_task_vector(base_model, task_vector, 2 * lambda_val, device)
        doubled_compositional = apply_task_vector_compositional(
            base_model, task_vector, lambda_val, device
        )

        # ─── Geometric distances (for reference) ───
        geom_dist_linear = compute_model_distance(doubled_linear, base_model)
        geom_dist_comp = compute_model_distance(doubled_compositional, base_model)

        # ─── Functional metrics ───
        intermediate_loss = evaluate_model(
            intermediate_model, tokenizer, general_eval_texts, device
        )

        # Linear version: M_base + 2λT
        doubled_loss_linear = evaluate_model(
            doubled_linear, tokenizer, general_eval_texts, device
        )
        functional_return_linear = abs(doubled_loss_linear - base_loss)

        # Compositional version: M_base + λT + λ²T
        doubled_loss_comp = evaluate_model(
            doubled_compositional, tokenizer, general_eval_texts, device
        )
        functional_return_comp = abs(doubled_loss_comp - base_loss)

        # ─── Task performance ───
        task_performance = evaluate_task_performance(
            intermediate_model, tokenizer, task_eval_texts, device
        )

        # ─── Utility scores ───
        utility_linear = functional_return_linear + 0.5 * abs(task_performance - base_loss)
        utility_comp = functional_return_comp + 0.5 * abs(task_performance - base_loss)

        result = InverseResult(
            lambda_val=lambda_val,
            geometric_distance_linear=geom_dist_linear,
            geometric_distance_compositional=geom_dist_comp,
            base_loss=base_loss,
            intermediate_loss=intermediate_loss,
            doubled_loss_linear=doubled_loss_linear,
            functional_return_linear=functional_return_linear,
            doubled_loss_compositional=doubled_loss_comp,
            functional_return_compositional=functional_return_comp,
            intermediate_task_performance=task_performance,
            utility_score_linear=utility_linear,
            utility_score_compositional=utility_comp,
        )
        results.append(result)

        print(f"→ Lin={functional_return_linear:.4f}, Comp={functional_return_comp:.4f}")

        # Clean up
        del intermediate_model, doubled_linear, doubled_compositional

        # Clear device cache (device-agnostic)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return results


def analyze_results(results: list[InverseResult]) -> dict:
    """Analyze results for both linear and compositional versions.

    Returns dict with best λ values for each metric.
    """

    # Sort by functional return for both versions
    sorted_by_linear = sorted(results, key=lambda r: r.functional_return_linear)
    sorted_by_comp = sorted(results, key=lambda r: r.functional_return_compositional)

    # Find special λ values (small functional return AND non-zero λ)
    special_linear = []
    special_comp = []

    print(f"\n{'='*70}")
    print("ANALYSIS: COMPARING LINEAR vs COMPOSITIONAL VERSIONS")
    print(f"{'='*70}\n")

    # ─── LINEAR VERSION ───
    print("=" * 70)
    print("LINEAR VERSION: M_base + 2λT")
    print("=" * 70)
    print("\nTop 10 λ values by functional return |L(M_base + 2λT) - L(M_base)|:\n")

    for i, result in enumerate(sorted_by_linear[:10], 1):
        print(f"{i:2d}. λ = {result.lambda_val:+.4f}")
        near_zero = '← NEAR ZERO!' if result.functional_return_linear < 0.1 else ''
        print(f"    Functional return (linear): {result.functional_return_linear:.6f} {near_zero}")
        print(f"    Doubled loss: {result.doubled_loss_linear:.4f}")
        print(f"    Base loss: {result.base_loss:.4f}")
        print(f"    Δ = {result.doubled_loss_linear - result.base_loss:+.4f}")

        if abs(result.lambda_val) > 0.1 and result.functional_return_linear < 1.0:
            special_linear.append(result)
            print("    ★ SPECIAL: Non-trivial functional self-inverse point!")
        print()

    # ─── COMPOSITIONAL VERSION ───
    print("\n" + "=" * 70)
    print("COMPOSITIONAL VERSION: M_base + λT + λ²T")
    print("=" * 70)
    print("\nTop 10 λ values by functional return |L(M_base + λT + λ²T) - L(M_base)|:\n")

    for i, result in enumerate(sorted_by_comp[:10], 1):
        print(f"{i:2d}. λ = {result.lambda_val:+.4f}")
        near_zero = '← NEAR ZERO!' if result.functional_return_compositional < 0.1 else ''
        print(
            f"    Functional return (comp): "
            f"{result.functional_return_compositional:.6f} {near_zero}"
        )
        print(f"    Doubled loss: {result.doubled_loss_compositional:.4f}")
        print(f"    Base loss: {result.base_loss:.4f}")
        print(f"    Δ = {result.doubled_loss_compositional - result.base_loss:+.4f}")

        if abs(result.lambda_val) > 0.1 and result.functional_return_compositional < 1.0:
            special_comp.append(result)
            print("    ★ SPECIAL: Non-trivial functional self-inverse point!")
        print()

    # ─── Comparison ───
    print("\n" + "=" * 70)
    print("GEOMETRIC DISTANCE VERIFICATION")
    print("=" * 70)
    sorted_by_geom_linear = sorted(results, key=lambda r: r.geometric_distance_linear)
    print(f"\nLinear: Minimum at λ = {sorted_by_geom_linear[0].lambda_val:.4f}")
    print(f"  Distance: {sorted_by_geom_linear[0].geometric_distance_linear:.6f}")
    print("  (Expected: ||M_base + 2λT - M_base|| = 2|λ|·||T|| minimized at λ=0)")

    sorted_by_geom_comp = sorted(results, key=lambda r: r.geometric_distance_compositional)
    print(f"\nCompositional: Minimum at λ = {sorted_by_geom_comp[0].lambda_val:.4f}")
    print(f"  Distance: {sorted_by_geom_comp[0].geometric_distance_compositional:.6f}")
    print("  (||M_base + λT + λ²T - M_base|| = |λ(1+λ)|·||T||, minimized near λ=0 or λ=-1)\n")

    return {
        'special_linear': special_linear,
        'special_compositional': special_comp,
        'sorted_by_linear': sorted_by_linear[:10],
        'sorted_by_compositional': sorted_by_comp[:10],
        'best_linear': sorted_by_linear[0],
        'best_compositional': sorted_by_comp[0],
        'all_results': results,
    }


def plot_results(
    results: list[InverseResult],
    analysis: dict,
    output_path: str = "self_inverse_task_vectors.png"
):
    """Visualize results comparing linear and compositional versions."""

    lambdas = [r.lambda_val for r in results]

    # Extract metrics
    geom_linear = [r.geometric_distance_linear for r in results]
    geom_comp = [r.geometric_distance_compositional for r in results]
    func_linear = [r.functional_return_linear for r in results]
    func_comp = [r.functional_return_compositional for r in results]
    int_losses = [r.intermediate_loss for r in results]
    dbl_linear = [r.doubled_loss_linear for r in results]
    dbl_comp = [r.doubled_loss_compositional for r in results]
    base_loss = results[0].base_loss
    task_perfs = [r.intermediate_task_performance for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        'Self-Inverse Task Vectors: Linear vs Compositional Doubling\n'
        'Inspired by Eckmann & Tlusty (2025)',
        fontsize=14, fontweight='bold', y=0.995
    )

    # ─── Plot 1: Geometric Distances (both versions) ───
    axes[0, 0].plot(
        lambdas, geom_linear, 'blue', linewidth=2,
        label='Linear: 2|λ|·||T||', alpha=0.7
    )
    axes[0, 0].plot(
        lambdas, geom_comp, 'red', linewidth=2,
        label='Comp: |λ(1+λ)|·||T||', alpha=0.7
    )
    axes[0, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('λ', fontsize=11)
    axes[0, 0].set_ylabel('Parameter Distance', fontsize=11)
    axes[0, 0].set_title('Geometric Distance (for reference)', fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # ─── Plot 2: Functional Returns (COMPARISON) ───
    axes[0, 1].plot(lambdas, func_linear, 'b-', linewidth=2.5, label='Linear: 2λT', alpha=0.8)
    axes[0, 1].plot(lambdas, func_comp, 'r-', linewidth=2.5, label='Comp: λT + λ²T', alpha=0.8)
    axes[0, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1.5)

    # Highlight special points
    if analysis['special_linear']:
        sp_lam = [r.lambda_val for r in analysis['special_linear']]
        sp_ret = [r.functional_return_linear for r in analysis['special_linear']]
        axes[0, 1].scatter(sp_lam, sp_ret, color='darkblue', s=120,
                          zorder=5, marker='*', edgecolors='black', linewidth=0.5)
    if analysis['special_compositional']:
        sp_lam = [r.lambda_val for r in analysis['special_compositional']]
        sp_ret = [r.functional_return_compositional for r in analysis['special_compositional']]
        axes[0, 1].scatter(sp_lam, sp_ret, color='darkred', s=120,
                          zorder=5, marker='*', edgecolors='black', linewidth=0.5)

    axes[0, 1].set_xlabel('λ', fontsize=11)
    axes[0, 1].set_ylabel('|L(M_doubled) - L(M_base)|', fontsize=11)
    axes[0, 1].set_title(
        'Functional Return (KEY METRIC)',
        fontsize=11, fontweight='bold', color='darkblue'
    )
    axes[0, 1].legend(fontsize=9, loc='best')
    axes[0, 1].grid(True, alpha=0.3)

    # ─── Plot 3: Loss Landscape ───
    axes[0, 2].plot(
        lambdas, int_losses, 'green', linewidth=2,
        label='M + λT (intermediate)', alpha=0.7
    )
    axes[0, 2].plot(
        lambdas, dbl_linear, 'blue', linewidth=2,
        label='M + 2λT (linear)', alpha=0.7
    )
    axes[0, 2].plot(
        lambdas, dbl_comp, 'red', linewidth=2,
        label='M + λT + λ²T (comp)', alpha=0.7
    )
    axes[0, 2].axhline(
        y=base_loss, color='black', linestyle='--',
        linewidth=2, label='Base', alpha=0.6
    )
    axes[0, 2].set_xlabel('λ', fontsize=11)
    axes[0, 2].set_ylabel('Loss', fontsize=11)
    axes[0, 2].set_title('Loss Landscape Comparison', fontsize=11, fontweight='bold')
    axes[0, 2].legend(fontsize=8, loc='best')
    axes[0, 2].grid(True, alpha=0.3)

    # ─── Plot 4: Functional Return vs Task Performance ───
    axes[1, 0].scatter(
        func_linear, task_perfs, c=lambdas, cmap='coolwarm',
        s=60, alpha=0.6, edgecolors='black', linewidth=0.5,
        marker='o', label='Linear'
    )
    axes[1, 0].scatter(
        func_comp, task_perfs, c=lambdas, cmap='coolwarm',
        s=60, alpha=0.6, edgecolors='black', linewidth=0.5, marker='^'
    )
    axes[1, 0].set_xlabel('Functional Return', fontsize=11)
    axes[1, 0].set_ylabel('Task Performance (Intermediate)', fontsize=11)
    axes[1, 0].set_title('Return vs Utility Trade-off', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('λ value', fontsize=9)

    # ─── Plot 5: Signed Functional Return Comparison ───
    deltas_linear = [r.doubled_loss_linear - r.base_loss for r in results]
    deltas_comp = [r.doubled_loss_compositional - r.base_loss for r in results]
    axes[1, 1].plot(lambdas, deltas_linear, 'b-', linewidth=2, label='Linear', alpha=0.7)
    axes[1, 1].plot(lambdas, deltas_comp, 'r-', linewidth=2, label='Compositional', alpha=0.7)
    axes[1, 1].axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    axes[1, 1].fill_between(lambdas, 0, deltas_linear, alpha=0.15, color='blue')
    axes[1, 1].fill_between(lambdas, 0, deltas_comp, alpha=0.15, color='red')
    axes[1, 1].set_xlabel('λ', fontsize=11)
    axes[1, 1].set_ylabel('L(M_doubled) - L(M_base)', fontsize=11)
    axes[1, 1].set_title('Signed Functional Return', fontsize=11, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    # ─── Plot 6: Utility Scores Comparison ───
    utility_linear = [r.utility_score_linear for r in results]
    utility_comp = [r.utility_score_compositional for r in results]
    axes[1, 2].plot(lambdas, utility_linear, 'b-', linewidth=2, label='Linear', alpha=0.7)
    axes[1, 2].plot(lambdas, utility_comp, 'r-', linewidth=2, label='Compositional', alpha=0.7)
    best_lin_idx = np.argmin(utility_linear)
    best_comp_idx = np.argmin(utility_comp)
    axes[1, 2].scatter([lambdas[best_lin_idx]], [utility_linear[best_lin_idx]],
                       color='darkblue', s=150, zorder=5, marker='D', edgecolors='black')
    axes[1, 2].scatter([lambdas[best_comp_idx]], [utility_comp[best_comp_idx]],
                       color='darkred', s=150, zorder=5, marker='D', edgecolors='black')
    axes[1, 2].set_xlabel('λ', fontsize=11)
    axes[1, 2].set_ylabel('Utility Score (lower = better)', fontsize=11)
    axes[1, 2].set_title('Combined Utility Score', fontsize=11, fontweight='bold')
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to {output_path}")
    plt.close()


def main():
    """Test for self-inverse task vectors in neural network parameter space.

    Compares two interpretations of "doubling" a task vector:
    1. LINEAR: M_base + 2λT (scalar multiplication)
    2. COMPOSITIONAL: M_base + λT + λ²T (iterative application - closer to paper)

    Searches for non-zero λ where functional return ≈ 0.
    """

    print(f"\n{'='*70}")
    print("SELF-INVERSE TASK VECTORS EXPERIMENT")
    print("Inspired by Eckmann & Tlusty (2025)")
    print(f"{'='*70}")
    print("""
RESEARCH QUESTION:
─────────────────
Do there exist non-zero λ values where "doubled" task vector transformations
functionally return to the base model's loss?

TWO APPROACHES:
──────────────
1. LINEAR (simpler): L(M_base + 2λT) ≈? L(M_base)
   → Tests for periodicity/symmetry in loss landscape

2. COMPOSITIONAL (closer to paper): L(M_base + λT + λ²T) ≈? L(M_base)
   → Mimics iterative application: apply λT, then apply induced change again
   → Analogous to W² = W ∘ W (composition) from rotation group

WHAT WE MEASURE:
───────────────
• GEOMETRIC: Parameter space distance (uninformative - minimized at λ=0)
• FUNCTIONAL: Loss landscape return |L(M_doubled) - L(M_base)| (KEY METRIC!)

If non-zero λ* exist with small functional return → suggests rich geometric
structure in loss landscapes, analogous to rotation group's 180° roots.
""")

    # Configuration
    model_name = "google/gemma-3-12b-it"

    # Device detection with MPS support for Apple Silicon
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    output_dir = "./outputs"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model: {model_name}")
    print(f"Device: {device}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("Loading base model...")
    device_map = get_device_map()
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )

    # For MPS/CPU, manually move to device if device_map is None
    if device_map is None and device != "cpu":
        base_model = base_model.to(device)  # type: ignore[assignment]

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

    # Fine-tune model on task
    print("Loading model for fine-tuning...")
    ft_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device_map
    )
    if device_map is None and device != "cpu":
        ft_model = ft_model.to(device)  # type: ignore[assignment]

    finetuned_model = fine_tune_model(
        base_model=ft_model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        output_dir=f"{output_dir}/finetuned_tmp",
        num_epochs=2,
        learning_rate=5e-5
    )

    # Compute task vector
    print("Computing task vector T = M_finetuned - M_base...")
    task_vector = compute_task_vector(base_model, finetuned_model)

    # Calculate task vector magnitude
    tv_magnitude = np.sqrt(sum((v**2).sum().item() for v in task_vector.values()))
    print(f"Task vector magnitude: ||T|| = {tv_magnitude:.4f}\n")

    # Run experiment
    results = find_inverse_lambda_values(
        base_model=base_model,
        task_vector=task_vector,
        tokenizer=tokenizer,
        general_eval_texts=general_eval_texts,
        task_eval_texts=task_eval_texts,
        lambda_range=(-2.0, 2.0),
        num_samples=50,
        device=device
    )

    # Analyze results
    analysis = analyze_results(results)

    # Plot results
    plot_results(results, analysis, f"{output_dir}/self_inverse_task_vectors.png")

    # Save detailed results
    results_path = f"{output_dir}/self_inverse_task_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'paper_reference': 'Eckmann & Tlusty (2025), arXiv:2502.14367v3',
                'description': 'Comparing linear vs compositional task vector doubling',
                'task': 'sentiment_analysis',
                'task_vector_magnitude': tv_magnitude,
                'model': model_name,
            },
            'methodology': {
                'linear_version': 'M_doubled = M_base + 2λT',
                'compositional_version': 'M_comp = M_base + λT + λ²T',
                'primary_metric': 'functional_return = |L(M_doubled) - L(M_base)|',
                'note': 'Compositional version closer to paper\'s W² = W ∘ W',
            },
            'best_results': {
                'linear': asdict(analysis['best_linear']),
                'compositional': asdict(analysis['best_compositional']),
            },
            'special_lambdas': {
                'linear': [asdict(r) for r in analysis['special_linear']],
                'compositional': [asdict(r) for r in analysis['special_compositional']],
            },
            'top_10': {
                'linear': [asdict(r) for r in analysis['sorted_by_linear']],
                'compositional': [asdict(r) for r in analysis['sorted_by_compositional']],
            },
        }, f, indent=2)
    print(f"📄 Results saved to {results_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    best_lin = analysis['best_linear']
    best_comp = analysis['best_compositional']

    print("══════════════════════════════════════")
    print("LINEAR VERSION (M_base + 2λT):")
    print("══════════════════════════════════════")
    print(f"  Best λ = {best_lin.lambda_val:+.4f}")
    print(f"  Functional return: {best_lin.functional_return_linear:.6f}")
    print(f"  Doubled loss: {best_lin.doubled_loss_linear:.4f}")
    print(f"  Base loss: {best_lin.base_loss:.4f}")
    print(f"  Δ = {best_lin.doubled_loss_linear - best_lin.base_loss:+.4f}")

    if analysis['special_linear']:
        print(f"\n  ✓ Found {len(analysis['special_linear'])} special non-zero λ values:")
        for i, sp in enumerate(analysis['special_linear'][:3], 1):
            print(f"     {i}. λ={sp.lambda_val:+.4f}, F_return={sp.functional_return_linear:.4f}")
    else:
        print("\n  ✗ No special non-zero λ values found")

    print("\n══════════════════════════════════════")
    print("COMPOSITIONAL VERSION (M_base + λT + λ²T):")
    print("══════════════════════════════════════")
    print(f"  Best λ = {best_comp.lambda_val:+.4f}")
    print(f"  Functional return: {best_comp.functional_return_compositional:.6f}")
    print(f"  Doubled loss: {best_comp.doubled_loss_compositional:.4f}")
    print(f"  Base loss: {best_comp.base_loss:.4f}")
    print(f"  Δ = {best_comp.doubled_loss_compositional - best_comp.base_loss:+.4f}")

    if analysis['special_compositional']:
        print(f"\n  ✓ Found {len(analysis['special_compositional'])} special non-zero λ values:")
        for i, sp in enumerate(analysis['special_compositional'][:3], 1):
            print(
                f"     {i}. λ={sp.lambda_val:+.4f}, "
                f"F_return={sp.functional_return_compositional:.4f}"
            )
    else:
        print("\n  ✗ No special non-zero λ values found")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print("""
CONNECTION TO PAPER:
───────────────────
Eckmann & Tlusty (2025) prove that for rotation groups SO(3)/SU(2), almost any
walk W can be scaled and doubled to return to identity: [W(λ)]² = I for abundant
λ values. Key insight: 180° rotations are common (f₁(π) = 2/π), and squaring
them gives identity (R(n,π)² = I).

OUR EXPERIMENT:
──────────────
We test whether neural network loss landscapes exhibit analogous "functional roots
of identity" under task vector transformations.

TWO VERSIONS TESTED:
  1. LINEAR: M_base + 2λT
     → Simpler, but 2λT is scalar multiplication, NOT composition
     → Tests for periodicity/symmetry in loss landscape

  2. COMPOSITIONAL: M_base + λT + λ²T
     → Closer to paper's W² = W ∘ W (apply, then apply induced change again)
     → Tests for iterative functional return

STRUCTURAL DIFFERENCES FROM PAPER:
──────────────────────────────────
⚠️  Rotations: MULTIPLICATIVE GROUP (W₁ ∘ W₂), identity I, doubling W²
⚠️  Task vectors: ADDITIVE VECTOR SPACE (v₁ + v₂), identity 0
⚠️  No Haar measure on task vectors → can't prove abundance of solutions
⚠️  This is an EMPIRICAL exploration, not a proven theorem

WHAT FINDINGS MEAN:
──────────────────
IF non-zero λ* found with small functional return:
  ✓ Suggests rich geometric structure in loss landscapes
  ✓ Special scaling factors for task vector arithmetic
  ✓ Potential for model merging, multi-task learning applications
  ✓ Hints at deeper algebraic structure in parameter space

IF no such λ* exist:
  ✓ Loss landscape is monotonic along task vector direction
  ✓ Task vectors lack symmetry properties of rotations
  ✓ Still provides insights into parameter space geometry

COMPOSITIONAL VERSION is conceptually closer to the paper's mathematics,
but both versions explore functional return in the loss landscape.
""")


if __name__ == "__main__":
    main()
