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
class AlphaSweepResult:
    """Store results for a specific α value along the line M(α) = M_base + αT.

    This directly answers: "What does the loss look like as we move along
    the task vector direction?"
    """
    alpha: float                    # Scaling factor α
    loss: float                     # L(M_base + αT)
    base_loss: float                # L(M_base) - reference point
    functional_return: float        # |L(M_base + αT) - L(M_base)|
    task_performance: float         # Task-specific loss at M_base + αT


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


def sweep_alpha_values(
    base_model: PreTrainedModel,
    task_vector: dict[str, torch.Tensor],
    tokenizer,
    general_eval_texts: list[str],
    task_eval_texts: list[str],
    alpha_range: tuple[float, float] = (-3.0, 3.0),
    num_samples: int = 100,
    device: str = "cuda"
) -> list[AlphaSweepResult]:
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

    print(f"\n{'='*70}")
    print("LOSS LANDSCAPE SWEEP: L(M_base + αT)")
    print(f"{'='*70}")
    print(f"Range: α ∈ [{alpha_range[0]:.1f}, {alpha_range[1]:.1f}]")
    print(f"Samples: {num_samples}")
    print("\nQuestion: Does the loss curve cross L(M_base) at any α ≠ 0?\n")

    alpha_values = np.linspace(alpha_range[0], alpha_range[1], num_samples)
    results = []

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
    base_loss = evaluate_model(base_model, tokenizer, general_eval_texts, device)
    print(f"Base model loss L(M_base): {base_loss:.4f}\n")

    for i, alpha in enumerate(alpha_values):
        print(f"[{i+1}/{num_samples}] α = {alpha:+.3f} ", end="", flush=True)

        # Modify model parameters in-place: M_alpha = M_base + αT
        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in task_vector:
                    # Set param to: original + α * task_vector
                    param.copy_(
                        original_params[name] + alpha * task_vector[name].to(param.device)
                    )

        # Evaluate L(M_alpha)
        loss_alpha = evaluate_model(base_model, tokenizer, general_eval_texts, device)

        # Compute functional return |L(M_alpha) - L(M_base)|
        functional_return = abs(loss_alpha - base_loss)

        # Task performance
        task_performance = evaluate_task_performance(
            base_model, tokenizer, task_eval_texts, device
        )

        result = AlphaSweepResult(
            alpha=alpha,
            loss=loss_alpha,
            base_loss=base_loss,
            functional_return=functional_return,
            task_performance=task_performance,
        )
        results.append(result)

        print(f"→ L(α)={loss_alpha:.4f}, |ΔL|={functional_return:.4f}")

    # Restore original parameters at the end
    print("\nRestoring base model parameters...")
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if name in original_params:
                param.copy_(original_params[name])

    return results


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
    print(f"  Δ from base = {min_task_result.task_performance - min_task_result.base_loss:+.4f}\n")

    # Best functional return (smallest |L(α) - L_base|)
    print("Best Functional Return (smallest |L(α) - L_base|):")
    for i, result in enumerate(sorted_by_return[:5], 1):
        print(f"  {i}. α = {result.alpha:+.4f}, |ΔL| = {result.functional_return:.6f}")

    # Zero-crossings
    print("\nZero-Crossings (where L(α) ≈ L_base for α ≠ 0):")
    if zero_crossings:
        print(f"  Found {len(zero_crossings)} crossing(s):")
        for i, result in enumerate(zero_crossings[:5], 1):
            print(f"  {i}. α = {result.alpha:+.4f}, L(α) = {result.loss:.4f}, "
                  f"|ΔL| = {result.functional_return:.6f} ★")
    else:
        print(f"  No zero-crossings found (threshold: |ΔL| < {threshold})")
        print("  → Loss is monotonic along task vector direction")

    return {
        'min_general_loss': min_general_result,
        'min_task_loss': min_task_result,
        'best_return': sorted_by_return[0],
        'zero_crossings': zero_crossings,
        'sorted_by_return': sorted_by_return[:10],
        'all_results': results,
    }


def plot_results(
    results: list[AlphaSweepResult],
    analysis: dict,
    output_path: str = "loss_landscape_sweep.png"
):
    """Visualize loss landscape L(M_base + αT) vs α."""

    alphas = [r.alpha for r in results]
    losses = [r.loss for r in results]
    base_loss = results[0].base_loss
    functional_returns = [r.functional_return for r in results]
    task_perfs = [r.task_performance for r in results]

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Task Vector Loss Landscape: L(M_base + αT)\n'
        'Inspired by Eckmann & Tlusty (2025)',
        fontsize=14, fontweight='bold', y=0.995
    )

    # ─── Plot 1: MAIN - Loss vs α (KEY PLOT!) ───
    axes[0, 0].plot(alphas, losses, 'b-', linewidth=2.5, label='General Loss', alpha=0.8)
    axes[0, 0].plot(alphas, task_perfs, 'g-', linewidth=2.0, label='Task Loss', alpha=0.6)
    axes[0, 0].axhline(
        y=base_loss, color='red', linestyle='--',
        linewidth=2, label='L(M_base)', alpha=0.7
    )
    axes[0, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.4)

    # Highlight zero-crossings
    if analysis['zero_crossings']:
        zc_alphas = [r.alpha for r in analysis['zero_crossings']]
        zc_losses = [r.loss for r in analysis['zero_crossings']]
        axes[0, 0].scatter(zc_alphas, zc_losses, color='orange', s=150,
                          zorder=5, marker='*', edgecolors='black', linewidth=1,
                          label='Zero-crossings')

    # Highlight general minimum
    min_gen_result = analysis['min_general_loss']
    axes[0, 0].scatter([min_gen_result.alpha], [min_gen_result.loss], color='blue', s=150,
                      zorder=5, marker='D', edgecolors='black', linewidth=1,
                      label=f'Min General (α={min_gen_result.alpha:.2f})')

    # Highlight task minimum
    min_task_result = analysis['min_task_loss']
    axes[0, 0].scatter([min_task_result.alpha], [min_task_result.task_performance], color='green', s=150,
                      zorder=5, marker='D', edgecolors='black', linewidth=1,
                      label=f'Min Task (α={min_task_result.alpha:.2f})')

    axes[0, 0].set_xlabel('α', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss Landscape (KEY PLOT)', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=8, loc='best')
    axes[0, 0].grid(True, alpha=0.3)

    # ─── Plot 2: Functional Return |L(α) - L_base| ───
    axes[0, 1].plot(alphas, functional_returns, 'r-', linewidth=2.5, alpha=0.8)
    axes[0, 1].axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    axes[0, 1].axvline(x=0, color='gray', linestyle='--', alpha=0.4)

    # Highlight zero-crossings
    if analysis['zero_crossings']:
        zc_alphas = [r.alpha for r in analysis['zero_crossings']]
        zc_returns = [r.functional_return for r in analysis['zero_crossings']]
        axes[0, 1].scatter(zc_alphas, zc_returns, color='green', s=150,
                          zorder=5, marker='*', edgecolors='black', linewidth=1)

    axes[0, 1].set_xlabel('α', fontsize=12)
    axes[0, 1].set_ylabel('|L(α) - L(M_base)|', fontsize=12)
    axes[0, 1].set_title('Functional Return', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # ─── Plot 3: Signed Delta L(α) - L_base ───
    deltas = [r.loss - r.base_loss for r in results]
    axes[1, 0].plot(alphas, deltas, 'b-', linewidth=2, alpha=0.8)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[1, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.4)
    axes[1, 0].fill_between(alphas, 0, deltas, alpha=0.2, color='blue')

    # Highlight zero-crossings
    if analysis['zero_crossings']:
        zc_alphas = [r.alpha for r in analysis['zero_crossings']]
        zc_deltas = [r.loss - r.base_loss for r in analysis['zero_crossings']]
        axes[1, 0].scatter(zc_alphas, zc_deltas, color='green', s=150,
                          zorder=5, marker='*', edgecolors='black', linewidth=1)

    axes[1, 0].set_xlabel('α', fontsize=12)
    axes[1, 0].set_ylabel('L(α) - L(M_base)', fontsize=12)
    axes[1, 0].set_title('Signed Loss Delta', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # ─── Plot 4: Task Performance vs α ───
    axes[1, 1].plot(alphas, task_perfs, 'g-', linewidth=2, alpha=0.8, label='Task Loss')
    axes[1, 1].axhline(y=base_loss, color='gray', linestyle='--', alpha=0.5, label='Base Loss')
    axes[1, 1].axvline(x=0, color='gray', linestyle='--', alpha=0.4)

    axes[1, 1].set_xlabel('α', fontsize=12)
    axes[1, 1].set_ylabel('Task-Specific Loss', fontsize=12)
    axes[1, 1].set_title('Task Performance', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to {output_path}")
    plt.close()


def main():
    """Sweep loss landscape along task vector direction.

    Directly visualizes L(M_base + αT) for α ∈ [-3, 3].
    Answers: Does the loss curve cross L(M_base) at any α ≠ 0?

    Inspired by Eckmann & Tlusty (2025)'s rotation group self-inverse walks.
    """

    print(f"\n{'='*70}")
    print("TASK VECTOR LOSS LANDSCAPE EXPERIMENT")
    print("Inspired by Eckmann & Tlusty (2025)")
    print(f"{'='*70}")
    print("""
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
    results = sweep_alpha_values(
        base_model=base_model,
        task_vector=task_vector,
        tokenizer=tokenizer,
        general_eval_texts=general_eval_texts,
        task_eval_texts=task_eval_texts,
        alpha_range=(-3.0, 3.0),
        num_samples=100,
        device=device
    )

    # Analyze results
    analysis = analyze_results(results)

    # Plot results
    plot_results(results, analysis, f"{output_dir}/loss_landscape_sweep.png")

    # Save detailed results
    results_path = f"{output_dir}/loss_landscape_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'paper_reference': 'Eckmann & Tlusty (2025), arXiv:2502.14367v3',
                'description': 'Loss landscape sweep along task vector direction',
                'task': 'sentiment_analysis',
                'task_vector_magnitude': tv_magnitude,
                'model': model_name,
                'alpha_range': [-3.0, 3.0],
                'num_samples': 100,
            },
            'methodology': {
                'experiment': 'M(α) = M_base + αT',
                'primary_metric': 'L(α) = loss of M(α)',
                'question': 'Does L(α) cross L(M_base) at any α ≠ 0?',
            },
            'key_results': {
                'min_general_loss': asdict(analysis['min_general_loss']),
                'min_task_loss': asdict(analysis['min_task_loss']),
                'best_return': asdict(analysis['best_return']),
                'zero_crossings': [asdict(r) for r in analysis['zero_crossings']],
            },
            'top_10_by_return': [asdict(r) for r in analysis['sorted_by_return']],
            'all_results': [asdict(r) for r in results],
        }, f, indent=2)
    print(f"📄 Results saved to {results_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    min_general = analysis['min_general_loss']
    min_task = analysis['min_task_loss']
    best_return = analysis['best_return']
    zero_crossings = analysis['zero_crossings']

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
            print(f"     {i}. α = {zc.alpha:+.4f}, L(α) = {zc.loss:.4f}, |ΔL| = {zc.functional_return:.6f}")
    else:
        print("  ✗ No zero-crossings found")
        print("  → Loss is monotonic along task vector direction")

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    print("""
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
""")


if __name__ == "__main__":
    main()
