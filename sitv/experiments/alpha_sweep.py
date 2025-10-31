"""
Alpha sweep experiment for SITV.

This module implements the 1D alpha sweep experiment that explores
the loss landscape along a task vector direction.
"""

import time
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any
from transformers import PreTrainedModel

from sitv.experiments.base import Experiment
from sitv.data.models import AlphaSweepResult
from sitv.core.evaluation import EvaluationService


class AlphaSweepExperiment(Experiment):
    """Alpha sweep experiment: L(M_base + α·T).

    This experiment sweeps over alpha values and evaluates the loss landscape
    along the task vector direction. It answers the question:
    "Does the loss curve cross L(M_base) at any α ≠ 0?"

    The experiment is memory-efficient, reusing a single model instance and
    modifying parameters in-place.

    Attributes:
        task_vector: Task vector T = M_finetuned - M_base
        general_eval_texts: Neutral evaluation texts
        task_eval_texts: Task-specific evaluation texts
        alpha_range: Range of α values to test
        num_samples: Number of α samples
        enable_squaring_test: Whether to test doubling (Enhancement #3)
        evaluator: Evaluation service instance
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        task_vector: Dict[str, torch.Tensor],
        tokenizer,
        general_eval_texts: List[str],
        task_eval_texts: List[str],
        general_eval_categories: List[str] = None,
        alpha_range: tuple[float, float] = (-3.0, 3.0),
        num_samples: int = 100,
        device: str = "cuda",
        enable_squaring_test: bool = True,
    ):
        """Initialize the alpha sweep experiment.

        Args:
            base_model: Base model M_base
            task_vector: Task vector T = M_finetuned - M_base
            tokenizer: Tokenizer for evaluation
            general_eval_texts: Neutral evaluation texts
            task_eval_texts: Task-specific evaluation texts
            general_eval_categories: Category labels for general_eval_texts (optional)
            alpha_range: Range of α values to test (min, max)
            num_samples: Number of α samples
            device: Device for computation
            enable_squaring_test: Whether to test M(2α) as well
        """
        super().__init__(base_model, tokenizer, device)
        self.task_vector = task_vector
        self.general_eval_texts = general_eval_texts
        self.general_eval_categories = general_eval_categories
        self.task_eval_texts = task_eval_texts
        self.alpha_range = alpha_range
        self.num_samples = num_samples
        self.enable_squaring_test = enable_squaring_test
        self.evaluator = EvaluationService(tokenizer, device)

    def run(self) -> tuple[List[AlphaSweepResult], Dict[str, Any]]:
        """Run the alpha sweep experiment.

        Returns:
            Tuple of (results_list, metadata_dict)
                - results_list: List of AlphaSweepResult objects
                - metadata_dict: Dictionary with timing and configuration info
        """
        self.start_timing()

        print(f"\n{'='*70}")
        print("LOSS LANDSCAPE SWEEP: L(M_base + αT)")
        print(f"{'='*70}")
        print(f"Range: α ∈ [{self.alpha_range[0]:.1f}, {self.alpha_range[1]:.1f}]")
        print(f"Samples: {self.num_samples}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nQuestion: Does the loss curve cross L(M_base) at any α ≠ 0?\n")

        # Prepare model
        self.prepare_model()

        # Clone original parameters
        original_params = self.clone_parameters(self.task_vector)

        # Compute base model loss
        print("Computing base model loss...")
        base_loss = self.evaluator.evaluate(self.base_model, self.general_eval_texts)
        print(f"Base model loss L(M_base): {base_loss:.4f}\n")

        # Generate alpha values
        alpha_values = np.linspace(
            self.alpha_range[0],
            self.alpha_range[1],
            self.num_samples
        )

        # Run sweep
        results = []
        alpha_times = []

        for i, alpha in enumerate(alpha_values):
            alpha_start = time.time()

            # Calculate and display progress
            progress_pct = (i / self.num_samples) * 100
            eta_str = self._calculate_eta(alpha_times, i)

            print(
                f"[{i+1:3d}/{self.num_samples}] ({progress_pct:5.1f}%) "
                f"α = {alpha:+.3f} | ",
                end="",
                flush=True
            )

            # Evaluate at alpha
            result = self._evaluate_at_alpha(alpha, original_params, base_loss)
            results.append(result)

            # Track timing
            alpha_elapsed = time.time() - alpha_start
            alpha_times.append(alpha_elapsed)

            # Print result
            self._print_result(result, alpha_elapsed, eta_str)

        # Restore parameters
        self.restore_parameters(original_params)

        self.end_timing()

        # Print summary
        self._print_summary(alpha_times)

        # Return results with metadata
        metadata = self._create_metadata(alpha_times)
        return results, metadata

    def _evaluate_at_alpha(
        self,
        alpha: float,
        original_params: Dict[str, torch.Tensor],
        base_loss: float
    ) -> AlphaSweepResult:
        """Evaluate model at a specific alpha value.

        Args:
            alpha: Alpha scaling factor
            original_params: Original model parameters
            base_loss: Base model loss for reference

        Returns:
            AlphaSweepResult with evaluation metrics
        """
        # Apply task vector: M_alpha = M_base + α·T
        self.apply_task_vector(original_params, self.task_vector, alpha)

        # Evaluate L(M_alpha)
        loss_alpha = self.evaluator.evaluate(
            self.base_model,
            self.general_eval_texts
        )

        # Compute functional return
        functional_return = abs(loss_alpha - base_loss)

        # Task performance
        task_performance = self.evaluator.evaluate_task_performance(
            self.base_model,
            self.task_eval_texts
        )

        # Squaring test (if enabled)
        loss_2alpha = 0.0
        functional_return_2alpha = 0.0

        if self.enable_squaring_test:
            loss_2alpha, functional_return_2alpha = self._evaluate_squaring_test(
                alpha,
                original_params,
                base_loss
            )

        # Per-category losses (if categories provided)
        category_losses = {}
        if self.general_eval_categories:
            category_losses = self.evaluator.evaluate_by_category(
                self.base_model,
                self.general_eval_texts,
                self.general_eval_categories
            )

        # Compute perplexities
        perplexity = np.exp(loss_alpha)
        perplexity_2alpha = np.exp(loss_2alpha) if self.enable_squaring_test else 0.0

        return AlphaSweepResult(
            alpha=alpha,
            loss=loss_alpha,
            base_loss=base_loss,
            functional_return=functional_return,
            task_performance=task_performance,
            loss_2alpha=loss_2alpha,
            functional_return_2alpha=functional_return_2alpha,
            perplexity=perplexity,
            perplexity_2alpha=perplexity_2alpha,
            category_losses=category_losses,
        )

    def _evaluate_squaring_test(
        self,
        alpha: float,
        original_params: Dict[str, torch.Tensor],
        base_loss: float
    ) -> tuple[float, float]:
        """Evaluate squaring test: M(2α) = M_base + 2α·T

        Args:
            alpha: Original alpha value
            original_params: Original model parameters
            base_loss: Base model loss

        Returns:
            Tuple of (loss_2alpha, functional_return_2alpha)
        """
        # Modify model to M_base + 2α·T
        self.apply_task_vector(original_params, self.task_vector, 2.0 * alpha)

        # Evaluate L(M_2alpha)
        loss_2alpha = self.evaluator.evaluate(
            self.base_model,
            self.general_eval_texts
        )
        functional_return_2alpha = abs(loss_2alpha - base_loss)

        # Restore to M_alpha for consistency
        self.apply_task_vector(original_params, self.task_vector, alpha)

        return loss_2alpha, functional_return_2alpha

    def _calculate_eta(self, alpha_times: List[float], current_idx: int) -> str:
        """Calculate ETA string.

        Args:
            alpha_times: List of times for completed alphas
            current_idx: Current iteration index

        Returns:
            Formatted ETA string
        """
        if alpha_times:
            avg_time = sum(alpha_times) / len(alpha_times)
            remaining = self.num_samples - current_idx
            return self.format_eta(avg_time, remaining)
        else:
            return "ETA: calculating..."

    def _print_result(
        self,
        result: AlphaSweepResult,
        elapsed: float,
        eta_str: str
    ) -> None:
        """Print result for current alpha.

        Args:
            result: AlphaSweepResult to print
            elapsed: Time elapsed for this alpha
            eta_str: ETA string
        """
        if self.enable_squaring_test:
            print(
                f"L(α)={result.loss:.4f}, L(2α)={result.loss_2alpha:.4f}, "
                f"|ΔL|={result.functional_return:.4f}, "
                f"|ΔL(2α)|={result.functional_return_2alpha:.4f} | "
                f"{elapsed:.1f}s | {eta_str}"
            )
        else:
            print(
                f"L(α)={result.loss:.4f}, "
                f"|ΔL|={result.functional_return:.4f} | "
                f"{elapsed:.1f}s | {eta_str}"
            )

    def _print_summary(self, alpha_times: List[float]) -> None:
        """Print experiment summary.

        Args:
            alpha_times: List of times for each alpha evaluation
        """
        duration = self.get_duration()
        avg_time = sum(alpha_times) / len(alpha_times) if alpha_times else 0.0

        print(f"\n{'='*70}")
        print("ALPHA SWEEP COMPLETE")
        print(f"{'='*70}")
        print(f"  Duration: {duration / 60:.1f} minutes ({duration:.0f}s)")
        print(f"  Samples: {self.num_samples}")
        print(f"  Avg time/sample: {avg_time:.2f}s")
        print(f"{'='*70}\n")

    def _create_metadata(self, alpha_times: List[float]) -> Dict[str, Any]:
        """Create metadata dictionary for results.

        Args:
            alpha_times: List of times for each alpha evaluation

        Returns:
            Metadata dictionary
        """
        metadata = self.get_timing_metadata()
        avg_time = sum(alpha_times) / len(alpha_times) if alpha_times else 0.0
        metadata["time_per_alpha_seconds"] = avg_time
        return metadata
