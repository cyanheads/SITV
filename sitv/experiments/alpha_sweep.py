"""
Alpha sweep experiment for SITV.

This module implements the 1D alpha sweep experiment that explores
the loss landscape along a task vector direction.
"""

import time
import numpy as np
import torch
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from transformers import PreTrainedModel

from sitv.experiments.base import Experiment
from sitv.experiments.config import SamplingConfig
from sitv.data.models import AlphaSweepResult
from sitv.core.evaluation import EvaluationService
from sitv.core.validation import validate_alpha_sweep_config, ValidationError
from sitv.core.error_handling import (
    retry_on_evaluation_failure,
    handle_evaluation_error,
    FailureTracker,
    safe_cuda_cleanup
)
from sitv.experiments.sampling import (
    UniformSampler,
    AdaptiveSampler,
    BayesianSampler
)

# Set up logger
logger = logging.getLogger(__name__)


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
        general_eval_categories: Optional[List[str]] = None,
        opposite_sentiment_eval_texts: Optional[List[str]] = None,
        alpha_range: tuple[float, float] = (-3.0, 3.0),
        num_samples: int = 100,
        device: str = "cuda",
        enable_squaring_test: bool = True,
        sampling_strategy: str = "uniform",
        sampling_config: Optional[SamplingConfig] = None,
    ):
        """Initialize the alpha sweep experiment.

        Args:
            base_model: Base model M_base
            task_vector: Task vector T = M_finetuned - M_base
            tokenizer: Tokenizer for evaluation
            general_eval_texts: Neutral evaluation texts
            task_eval_texts: Task-specific evaluation texts
            general_eval_categories: Category labels for general_eval_texts (optional)
            opposite_sentiment_eval_texts: Opposite sentiment texts for preference calculation (optional)
            alpha_range: Range of α values to test (min, max)
            num_samples: Number of α samples
            device: Device for computation
            enable_squaring_test: Whether to test M(2α) as well
            sampling_strategy: Sampling strategy ("uniform", "adaptive", "bayesian")
            sampling_config: Sampling configuration (adaptive/bayesian parameters)

        Raises:
            ValidationError: If any configuration parameter is invalid
        """
        super().__init__(base_model, tokenizer, device)

        # Validate all inputs before proceeding
        validate_alpha_sweep_config(
            alpha_range=alpha_range,
            num_samples=num_samples,
            general_eval_texts=general_eval_texts,
            task_eval_texts=task_eval_texts,
            task_vector=task_vector,
            general_eval_categories=general_eval_categories,
        )

        self.task_vector = task_vector
        self.general_eval_texts = general_eval_texts
        self.general_eval_categories = general_eval_categories
        self.task_eval_texts = task_eval_texts
        self.opposite_sentiment_eval_texts = opposite_sentiment_eval_texts
        self.alpha_range = alpha_range
        self.num_samples = num_samples
        self.enable_squaring_test = enable_squaring_test
        self.sampling_strategy = sampling_strategy.lower()
        self.sampling_config = sampling_config or SamplingConfig()
        self.evaluator = EvaluationService(tokenizer, device)
        self.failure_tracker = FailureTracker(
            max_consecutive_failures=5,
            max_total_failures_pct=0.3
        )

        # Create sampler based on strategy
        self.sampler = self._create_sampler()

    def _create_sampler(self):
        """Create the appropriate sampler based on strategy.

        Returns:
            Sampler instance (UniformSampler, AdaptiveSampler, or BayesianSampler)

        Raises:
            ValueError: If sampling strategy is invalid
        """
        if self.sampling_strategy == "uniform":
            return UniformSampler(
                alpha_range=self.alpha_range,
                num_samples=self.num_samples
            )
        elif self.sampling_strategy == "adaptive":
            return AdaptiveSampler(
                alpha_range=self.alpha_range,
                num_samples=self.num_samples,
                coarse_samples=self.sampling_config.adaptive_coarse_samples,
                refine_factor=self.sampling_config.adaptive_refine_factor,
                curvature_threshold=self.sampling_config.adaptive_curvature_threshold,
            )
        elif self.sampling_strategy == "bayesian":
            return BayesianSampler(
                alpha_range=self.alpha_range,
                num_samples=self.num_samples,
                n_initial=self.sampling_config.bayesian_n_initial,
                acquisition=self.sampling_config.bayesian_acquisition,
            )
        else:
            raise ValueError(
                f"Invalid sampling strategy: '{self.sampling_strategy}'. "
                f"Must be one of: 'uniform', 'adaptive', 'bayesian'"
            )

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

        # Pre-load task vector to device for performance
        device_task_vector = self.preload_task_vector_to_device(self.task_vector)

        try:
            # Compute base model loss
            print("Computing base model loss...")
            base_loss = self._evaluate_base_loss_with_retry()
            print(f"Base model loss L(M_base): {base_loss:.4f}\n")

            # Generate alpha values using configured sampler
            print(f"Sampling strategy: {self.sampling_strategy}")
            alpha_values = self.sampler.generate_samples()
            print(f"Generated {len(alpha_values)} alpha values\n")

            # Run sweep
            results = []
            alpha_times: list[float] = []
            total_alphas = len(alpha_values)

            for i, alpha in enumerate(alpha_values):
                alpha_start = time.time()

                # Calculate and display progress
                progress_pct = ((i + 1) / total_alphas) * 100
                eta_str = self._calculate_eta(alpha_times, i, total_alphas)

                print(
                    f"[{i+1:3d}/{total_alphas}] ({progress_pct:5.1f}%) "
                    f"α = {alpha:+.3f} | ",
                    end="",
                    flush=True
                )

                # Evaluate at alpha with error handling
                result = self._evaluate_at_alpha_safe(
                    alpha, original_params, device_task_vector, base_loss
                )

                if result is not None:
                    results.append(result)
                    self.failure_tracker.record_success(alpha)
                else:
                    # Evaluation failed, check if we should abort
                    should_abort, reason = self.failure_tracker.should_abort()
                    if should_abort:
                        logger.error(f"Aborting experiment: {reason}")
                        print(f"\n⚠️  Experiment aborted: {reason}")
                        break

                # Track timing
                alpha_elapsed = time.time() - alpha_start
                alpha_times.append(alpha_elapsed)

                # Print result
                if result is not None:
                    self._print_result(result, alpha_elapsed, eta_str)
                else:
                    print(f"FAILED | {alpha_elapsed:.1f}s | {eta_str}")

        finally:
            # Always restore parameters, even if experiment fails
            self.restore_parameters(original_params)
            safe_cuda_cleanup()

        self.end_timing()

        # Print summary
        self._print_summary(alpha_times, results, len(alpha_values))

        # Return results with metadata
        metadata = self._create_metadata(alpha_times)
        return results, metadata

    @retry_on_evaluation_failure(max_retries=2, return_on_failure=None)
    def _evaluate_base_loss_with_retry(self) -> float:
        """Evaluate base model loss with retry logic.

        Returns:
            Base model loss

        Raises:
            Exception: If evaluation fails after retries
        """
        return self.evaluator.evaluate(self.base_model, self.general_eval_texts)

    def _evaluate_at_alpha_safe(
        self,
        alpha: float,
        original_params: Dict[str, torch.Tensor],
        device_task_vector: Dict[str, torch.Tensor],
        base_loss: float
    ) -> AlphaSweepResult | None:
        """Safely evaluate at alpha with error handling.

        Args:
            alpha: Alpha scaling factor
            original_params: Original model parameters
            device_task_vector: Task vector pre-loaded to device
            base_loss: Base model loss for reference

        Returns:
            AlphaSweepResult or None if evaluation failed
        """
        try:
            return self._evaluate_at_alpha(alpha, original_params, device_task_vector, base_loss)
        except Exception as e:
            # Log the error and record failure
            logger.error(f"Evaluation failed at α={alpha:.3f}: {str(e)}")
            self.failure_tracker.record_failure(alpha, e)

            # Try to handle the error gracefully
            fallback_loss, should_continue = handle_evaluation_error(e, alpha, "alpha sweep")

            if not should_continue:
                # Critical error, return None to signal failure
                return None

            # Return a fallback result with inf loss
            return AlphaSweepResult(
                alpha=alpha,
                loss=fallback_loss,
                base_loss=base_loss,
                functional_return=abs(fallback_loss - base_loss),
                task_eval_loss=fallback_loss,
                loss_2alpha=0.0,
                functional_return_2alpha=0.0,
                perplexity=float('inf'),
                perplexity_2alpha=0.0,
                category_losses={},
            )

    def _evaluate_at_alpha(
        self,
        alpha: float,
        original_params: Dict[str, torch.Tensor],
        device_task_vector: Dict[str, torch.Tensor],
        base_loss: float
    ) -> AlphaSweepResult:
        """Evaluate model at a specific alpha value.

        Args:
            alpha: Alpha scaling factor
            original_params: Original model parameters
            device_task_vector: Task vector pre-loaded to device
            base_loss: Base model loss for reference

        Returns:
            AlphaSweepResult with evaluation metrics
        """
        # Apply task vector: M_alpha = M_base + α·T
        self.apply_task_vector(original_params, device_task_vector, alpha)

        # Evaluate L(M_alpha)
        loss_alpha = self.evaluator.evaluate(
            self.base_model,
            self.general_eval_texts
        )

        # Compute functional return
        functional_return = abs(loss_alpha - base_loss)

        # Task evaluation loss
        task_eval_loss = self.evaluator.evaluate_task_performance(
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
                device_task_vector,
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

        # Sentiment preference (if opposite sentiment texts provided)
        sentiment_preference = 0.0
        task_eval_loss_negative = 0.0
        if self.opposite_sentiment_eval_texts:
            _, task_eval_loss_negative, sentiment_preference = \
                self.evaluator.evaluate_sentiment_preference(
                    self.base_model,
                    self.task_eval_texts,
                    self.opposite_sentiment_eval_texts
                )

        return AlphaSweepResult(
            alpha=alpha,
            loss=loss_alpha,
            base_loss=base_loss,
            functional_return=functional_return,
            task_eval_loss=task_eval_loss,
            loss_2alpha=loss_2alpha,
            functional_return_2alpha=functional_return_2alpha,
            perplexity=perplexity,
            perplexity_2alpha=perplexity_2alpha,
            sentiment_preference=sentiment_preference,
            task_eval_loss_negative=task_eval_loss_negative,
            category_losses=category_losses,
        )

    def _evaluate_squaring_test(
        self,
        alpha: float,
        original_params: Dict[str, torch.Tensor],
        device_task_vector: Dict[str, torch.Tensor],
        base_loss: float
    ) -> tuple[float, float]:
        """Evaluate squaring test: M(2α) = M_base + 2α·T

        Args:
            alpha: Original alpha value
            original_params: Original model parameters
            device_task_vector: Task vector pre-loaded to device
            base_loss: Base model loss

        Returns:
            Tuple of (loss_2alpha, functional_return_2alpha)
        """
        # Modify model to M_base + 2α·T
        self.apply_task_vector(original_params, device_task_vector, 2.0 * alpha)

        # Evaluate L(M_2alpha)
        loss_2alpha = self.evaluator.evaluate(
            self.base_model,
            self.general_eval_texts
        )
        functional_return_2alpha = abs(loss_2alpha - base_loss)

        # Restore to M_alpha for consistency
        self.apply_task_vector(original_params, device_task_vector, alpha)

        return loss_2alpha, functional_return_2alpha

    def _calculate_eta(self, alpha_times: List[float], current_idx: int, total_samples: int) -> str:
        """Calculate ETA string.

        Args:
            alpha_times: List of times for completed alphas
            current_idx: Current iteration index
            total_samples: Total number of samples to evaluate

        Returns:
            Formatted ETA string
        """
        if alpha_times:
            avg_time = sum(alpha_times) / len(alpha_times)
            remaining = total_samples - current_idx
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

    def _print_summary(self, alpha_times: List[float], results: List[AlphaSweepResult], total_samples: int) -> None:
        """Print experiment summary.

        Args:
            alpha_times: List of times for each alpha evaluation
            results: List of results collected
            total_samples: Total number of samples evaluated
        """
        duration = self.get_duration()
        avg_time = sum(alpha_times) / len(alpha_times) if alpha_times else 0.0

        print(f"\n{'='*70}")
        print("ALPHA SWEEP COMPLETE")
        print(f"{'='*70}")
        print(f"  Duration: {duration / 60:.1f} minutes ({duration:.0f}s)")
        print(f"  Samples Completed: {len(results)}/{total_samples}")
        print(f"  Avg time/sample: {avg_time:.2f}s")

        # Print failure summary if there were any failures
        failure_summary = self.failure_tracker.get_summary()
        if "Failures: 0/" not in failure_summary:
            print(f"  {failure_summary}")

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
