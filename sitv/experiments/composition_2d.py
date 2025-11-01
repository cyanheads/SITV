"""
2D composition experiment for SITV.

This module implements the 2D task vector composition experiment that explores
loss landscapes under multi-task composition.
"""

import time
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any
from transformers import PreTrainedModel

from sitv.experiments.base import Experiment
from sitv.data.models import TwoDSweepResult
from sitv.core.evaluation import EvaluationService


class Composition2DExperiment(Experiment):
    """2D composition experiment: L(M_base + α·T1 + β·T2).

    This experiment sweeps over a 2D grid of (α, β) values to explore
    loss landscapes under task vector composition. It tests whether
    rotation-like patterns emerge when combining two task vectors.

    Attributes:
        task_vector_1: First task vector T1
        task_vector_2: Second task vector T2
        general_eval_texts: Evaluation texts
        alpha_range: Range of α values
        beta_range: Range of β values
        num_samples_per_dim: Samples per dimension (creates n²grid)
        evaluator: Evaluation service instance
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        task_vector_1: Dict[str, torch.Tensor],
        task_vector_2: Dict[str, torch.Tensor],
        tokenizer,
        general_eval_texts: List[str],
        alpha_range: tuple[float, float] = (-2.0, 2.0),
        beta_range: tuple[float, float] = (-2.0, 2.0),
        num_samples_per_dim: int = 20,
        device: str = "cuda",
        eval_batch_size: int = 8,
        eval_enable_mixed_precision: bool = True,
        eval_max_length: int = 512,
    ):
        """Initialize the 2D composition experiment.

        Args:
            base_model: Base model M_base
            task_vector_1: First task vector T1
            task_vector_2: Second task vector T2
            tokenizer: Tokenizer for evaluation
            general_eval_texts: Evaluation texts
            alpha_range: Range for α (T1 scaling)
            beta_range: Range for β (T2 scaling)
            num_samples_per_dim: Samples per dimension (total = num²)
            device: Device for computation
            eval_batch_size: Batch size for evaluation (default: 8)
            eval_enable_mixed_precision: Use FP16/BF16 for evaluation (default: True)
            eval_max_length: Max sequence length for evaluation (default: 512)
        """
        super().__init__(base_model, tokenizer, device)
        self.task_vector_1 = task_vector_1
        self.task_vector_2 = task_vector_2
        self.general_eval_texts = general_eval_texts
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.num_samples_per_dim = num_samples_per_dim
        self.evaluator = EvaluationService(
            tokenizer,
            device,
            batch_size=eval_batch_size,
            enable_mixed_precision=eval_enable_mixed_precision,
            max_length=eval_max_length
        )

    def run(self) -> tuple[List[TwoDSweepResult], Dict[str, Any]]:
        """Run the 2D composition experiment.

        Returns:
            Tuple of (results_list, metadata_dict)
                - results_list: List of TwoDSweepResult objects
                - metadata_dict: Dictionary with timing and configuration info
        """
        self.start_timing()

        total_evals = self.num_samples_per_dim ** 2

        print(f"\n{'='*70}")
        print("2D TASK VECTOR COMPOSITION: L(M_base + α·T1 + β·T2)")
        print(f"{'='*70}")
        print(f"α range: [{self.alpha_range[0]:.1f}, {self.alpha_range[1]:.1f}]")
        print(f"β range: [{self.beta_range[0]:.1f}, {self.beta_range[1]:.1f}]")
        print(f"Grid: {self.num_samples_per_dim}×{self.num_samples_per_dim} = {total_evals} evaluations")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nQuestion: Do we see rotation-like patterns under composition?\n")

        # Prepare model
        self.prepare_model()

        # Clone original parameters (both task vectors must cover same params)
        original_params = self._clone_2d_parameters()

        # Pre-load task vectors to device for performance and correctness
        print("Pre-loading task vector 1 to device...")
        self.task_vector_1 = self.preload_task_vector_to_device(self.task_vector_1)
        print("Pre-loading task vector 2 to device...")
        self.task_vector_2 = self.preload_task_vector_to_device(self.task_vector_2)

        # Compute base model loss
        print("Computing base model loss...")
        base_loss = self.evaluator.evaluate(self.base_model, self.general_eval_texts)
        print(f"Base model loss L(M_base): {base_loss:.4f}\n")

        # Generate grid values
        alpha_values = np.linspace(
            self.alpha_range[0],
            self.alpha_range[1],
            self.num_samples_per_dim
        )
        beta_values = np.linspace(
            self.beta_range[0],
            self.beta_range[1],
            self.num_samples_per_dim
        )

        # Run sweep
        results = []
        eval_times: list[float] = []
        eval_count = 0

        for i, alpha in enumerate(alpha_values):
            for j, beta in enumerate(beta_values):
                eval_start = time.time()
                eval_count += 1

                # Calculate and display progress
                progress_pct = (eval_count / total_evals) * 100
                eta_str = self._calculate_eta(eval_times, eval_count, total_evals)

                print(
                    f"[{eval_count:4d}/{total_evals}] ({progress_pct:5.1f}%) "
                    f"α={alpha:+.2f}, β={beta:+.2f} | ",
                    end="",
                    flush=True
                )

                # Evaluate at (alpha, beta)
                result = self._evaluate_at_alpha_beta(
                    alpha,
                    beta,
                    original_params,
                    base_loss
                )
                results.append(result)

                # Track timing
                eval_elapsed = time.time() - eval_start
                eval_times.append(eval_elapsed)

                # Print result
                print(
                    f"L={result.loss:.4f}, |ΔL|={result.functional_return:.4f} | "
                    f"{eval_elapsed:.1f}s | {eta_str}"
                )

        # Restore parameters
        self.restore_parameters(original_params)

        self.end_timing()

        # Print summary
        self._print_summary(eval_times, total_evals)

        # Return results with metadata
        metadata = self._create_metadata(eval_times, total_evals)
        return results, metadata

    def _clone_2d_parameters(self) -> Dict[str, torch.Tensor]:
        """Clone parameters for 2D composition.

        Returns:
            Dictionary of cloned parameters
        """
        print("Cloning base model parameters...")
        original_params = {}
        for name, param in self.base_model.named_parameters():
            if name in self.task_vector_1 and name in self.task_vector_2:
                original_params[name] = param.clone().detach()
        return original_params

    def _evaluate_at_alpha_beta(
        self,
        alpha: float,
        beta: float,
        original_params: Dict[str, torch.Tensor],
        base_loss: float
    ) -> TwoDSweepResult:
        """Evaluate model at a specific (alpha, beta) point.

        Args:
            alpha: Alpha scaling factor for T1
            beta: Beta scaling factor for T2
            original_params: Original model parameters
            base_loss: Base model loss for reference

        Returns:
            TwoDSweepResult with evaluation metrics
        """
        # Apply 2D composition: M(α,β) = M_base + α·T1 + β·T2
        self.apply_2d_composition(
            original_params,
            self.task_vector_1,
            self.task_vector_2,
            alpha,
            beta
        )

        # Evaluate L(M_alpha_beta)
        loss = self.evaluator.evaluate(self.base_model, self.general_eval_texts)

        # Compute metrics
        functional_return = abs(loss - base_loss)
        perplexity = np.exp(loss)

        return TwoDSweepResult(
            alpha=alpha,
            beta=beta,
            loss=loss,
            base_loss=base_loss,
            functional_return=functional_return,
            perplexity=perplexity,
        )

    def _calculate_eta(
        self,
        eval_times: List[float],
        current_count: int,
        total_count: int
    ) -> str:
        """Calculate ETA string.

        Args:
            eval_times: List of times for completed evaluations
            current_count: Current evaluation count
            total_count: Total evaluations to perform

        Returns:
            Formatted ETA string
        """
        if eval_times:
            avg_time = sum(eval_times) / len(eval_times)
            remaining = total_count - current_count
            return self.format_eta(avg_time, remaining)
        else:
            return "ETA: calculating..."

    def _print_summary(self, eval_times: List[float], total_evals: int) -> None:
        """Print experiment summary.

        Args:
            eval_times: List of times for each evaluation
            total_evals: Total number of evaluations performed
        """
        duration = self.get_duration()
        avg_time = sum(eval_times) / len(eval_times) if eval_times else 0.0

        print(f"\n{'='*70}")
        print("2D COMPOSITION SWEEP COMPLETE")
        print(f"{'='*70}")
        print(f"  Duration: {duration / 60:.1f} minutes ({duration:.0f}s)")
        print(f"  Evaluations: {total_evals}")
        print(f"  Avg time/eval: {avg_time:.2f}s")
        print(f"{'='*70}\n")

    def _create_metadata(
        self,
        eval_times: List[float],
        total_evals: int
    ) -> Dict[str, Any]:
        """Create metadata dictionary for results.

        Args:
            eval_times: List of times for each evaluation
            total_evals: Total number of evaluations

        Returns:
            Metadata dictionary
        """
        metadata = self.get_timing_metadata()
        avg_time = sum(eval_times) / len(eval_times) if eval_times else 0.0
        metadata["time_per_eval_seconds"] = avg_time
        metadata["total_evaluations"] = total_evals
        return metadata
