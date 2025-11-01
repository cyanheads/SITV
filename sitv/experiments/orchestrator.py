"""
Experiment orchestrator for SITV.

This module provides the ExperimentOrchestrator that coordinates the entire
experimental workflow from model loading to result generation.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sitv.analysis import ResultAnalyzer
from sitv.core import TaskVectorService, get_device_string
from sitv.data.models import ExperimentMetrics, ThreeDSweepResult, TwoDSweepResult
from sitv.data.tasks import get_predefined_tasks
from sitv.experiments import AlphaSweepExperiment, Composition2DExperiment
from sitv.experiments.composition_3d import Composition3DExperiment
from sitv.experiments.config import ExperimentConfig
from sitv.io import FileManager, PathManager
from sitv.models import FineTuner, ModelService
from sitv.reporting import MarkdownReportGenerator
from sitv.utils import print_banner, print_separator
from sitv.visualization import ResultPlotter


def set_random_seed(seed: int | None) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value. If None, uses non-deterministic behavior.
    """
    if seed is None:
        return

    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    # If you need strict determinism (may limit ops):
    # torch.use_deterministic_algorithms(True)


class ExperimentOrchestrator:
    """Orchestrator for SITV experiments.

    This orchestrator coordinates the entire experiment workflow:
    1. Load or fine-tune models
    2. Compute task vectors
    3. Run experiments (alpha sweep, 2D composition, 3D composition)
    4. Analyze results
    5. Generate reports and visualizations
    6. Save all outputs
    """

    def __init__(self, config: ExperimentConfig):
        """Initialize the experiment orchestrator.

        Args:
            config: Experiment configuration.
        """
        self.config = config

        # Set random seed for reproducibility
        set_random_seed(config.seed)

        self.metrics = ExperimentMetrics(start_time=datetime.now().isoformat())

        # Initialize services and paths
        self.device = config.device if config.device else get_device_string()
        device_map = None if config.device else "auto"

        self.model_service = ModelService(device_map=device_map)
        self.task_vector_service = TaskVectorService()
        self.file_manager = FileManager(config.output_dir)
        self.path_manager = PathManager(config.output_dir)

        # Ensure output directory exists
        self.path_manager.ensure_output_dir()

        # Store metrics metadata
        self.metrics.model_name = config.model_name
        self.metrics.device = self.device
        self.metrics.task_name = config.task_name
        self.metrics.general_eval_dataset = config.evaluation.general_dataset

        # Optional experiment outputs
        self.results_2d: list[TwoDSweepResult] | None = None
        self.results_3d: list[ThreeDSweepResult] | None = None
        self.composition_analysis: dict[str, Any] | None = None

    def run(self) -> None:
        """Run the complete experiment workflow."""
        print_banner("SITV EXPERIMENT START")
        print(f"Model: {self.config.model_name}")
        print(f"Task: {self.config.task_name}")
        print(f"General Eval Dataset: {self.config.evaluation.general_dataset}")
        print(f"Device: {self.device}")
        print(f"Output: {self.config.output_dir}")
        print(f"Analysis Only: {self.config.analysis_only}")

        # Fine-tuning configuration
        if not self.config.analysis_only:
            print("\nFine-Tuning:")
            print(f"  Epochs: {self.config.fine_tuning.num_epochs}")
            print(f"  Learning rate: {self.config.fine_tuning.learning_rate:.2e}")
            print(f"  Batch size: {self.config.fine_tuning.batch_size}")
            print(f"  Max length: {self.config.fine_tuning.max_length}")
            print(f"  Data repetition: {self.config.fine_tuning.data_repetition_factor}x")

        # Alpha sweep configuration
        print("\nAlpha Sweep:")
        print(
            f"  Range: [{self.config.alpha_sweep.alpha_range[0]:.1f}, {self.config.alpha_sweep.alpha_range[1]:.1f}]"
        )
        print(f"  Samples: {self.config.alpha_sweep.num_samples}")
        print(f"  Squaring test: {self.config.alpha_sweep.enable_squaring_test}")

        # Evaluation optimization settings
        print("\nEvaluation Performance:")
        print(f"  Batch size: {self.config.evaluation.batch_size}")
        print(f"  Mixed precision: {self.config.evaluation.enable_mixed_precision}")
        print(f"  Max length: {self.config.evaluation.max_length}")

        # 2D composition configuration
        if self.config.enable_2d_composition:
            print("\n2D Composition:")
            print(f"  Alpha range: {self.config.composition_2d.alpha_range}")
            print(f"  Beta range: {self.config.composition_2d.beta_range}")
            print(
                f"  Grid size: {self.config.composition_2d.num_samples_per_dim}×{self.config.composition_2d.num_samples_per_dim}"
            )

        # 3D composition configuration
        if self.config.enable_3d_composition:
            print("\n3D Composition:")
            print(
                f"  Tasks: {self.config.composition_3d.task_1}, {self.config.composition_3d.task_2}, {self.config.composition_3d.task_3}"
            )
            print(f"  Alpha range: {self.config.composition_3d.alpha_range}")
            print(f"  Beta range: {self.config.composition_3d.beta_range}")
            print(f"  Gamma range: {self.config.composition_3d.gamma_range}")
            print(
                f"  Grid size: {self.config.composition_3d.num_samples_per_dim}³ = {self.config.composition_3d.num_samples_per_dim**3} evaluations"
            )

        # Riemannian geometry configuration
        if self.config.geometry.enabled:
            print("\nRiemannian Geometry: ENABLED ★")
            print(f"  Metric type: {self.config.geometry.metric_type.value}")
            print(f"  Geodesic integration: {self.config.geometry.geodesic_integration.enabled}")
            if self.config.geometry.geodesic_integration.enabled:
                print(f"    RK4 steps: {self.config.geometry.geodesic_integration.num_steps}")
            print(f"  Fisher samples: {self.config.geometry.fisher_approximation.num_samples:,}")
            print(f"  Cache metric: {self.config.geometry.cache_metric}")
            if self.config.geometry.symmetry_analysis.enabled:
                print("  Symmetry analysis: ENABLED")
            if self.config.geometry.curvature_analysis.enabled:
                print("  Curvature analysis: ENABLED")
        else:
            print("\nRiemannian Geometry: Disabled (using Euclidean)")

        print_separator()
        print()

        # Phase 1: Load or fine-tune models for the main task vector (T1)
        if self.config.analysis_only:
            base_model, finetuned_model, tokenizer = self._load_saved_models()
        else:
            base_model, finetuned_model, tokenizer = self._fine_tune_and_save()

        # Phase 2: Compute task vector (T1)
        task_vector = self._compute_task_vector(base_model, finetuned_model)

        # Phase 3: Run alpha sweep experiment on T1
        results, analysis = self._run_alpha_sweep(base_model, task_vector, tokenizer)

        # Phase 4: Run 2D composition (if enabled and not analysis_only)
        if self.config.enable_2d_composition:
            if self.config.analysis_only:
                print("\n⚠️  Skipping 2D composition: analysis_only=True (requires precomputed T2).")
            else:
                self._run_2d_composition(base_model, task_vector, tokenizer)

        # Phase 5: Run 3D composition (if enabled and not analysis_only)
        if self.config.enable_3d_composition:
            if self.config.analysis_only:
                print(
                    "\n⚠️  Skipping 3D composition: analysis_only=True (requires precomputed T2/T3)."
                )
            else:
                self._run_3d_composition(base_model, task_vector, tokenizer)

        # Finalize metrics
        self._finalize_metrics()

        # Phase 6: Generate outputs
        self._generate_outputs(results, analysis)

        print_banner("SITV EXPERIMENT COMPLETE")
        print(f"Duration: {self.metrics.duration_seconds / 60:.1f} minutes")
        print(f"Output directory: {self.config.output_dir}")
        print_separator()
        print()

    def _load_saved_models(self):
        """Load previously saved models for analysis.

        Returns:
            Tuple of (base_model, finetuned_model, tokenizer)
        """
        print_banner("LOADING SAVED MODELS")

        # Check if models exist
        if not self.model_service.check_saved_models_exist(self.config.output_dir):
            raise FileNotFoundError(
                f"Saved models not found in {self.config.output_dir}. "
                "Run without --analysis-only first to fine-tune models."
            )

        # Load models
        base_model, finetuned_model = self.model_service.load_saved_models(
            self.config.output_dir, self.device
        )

        # Load tokenizer
        tokenizer = self.model_service.load_tokenizer(self.config.model_name)

        # Update metrics
        self.metrics.model_parameters = self.model_service.count_parameters(base_model)

        return base_model, finetuned_model, tokenizer

    def _fine_tune_and_save(self):
        """Fine-tune model on the main task and save for future analysis.

        Returns:
            Tuple of (base_model_reloaded, finetuned_model, tokenizer)
        """
        print_banner("MODEL FINE-TUNING")

        # Load base model and tokenizer
        base_model, tokenizer = self.model_service.load_model_and_tokenizer(
            self.config.model_name, self.device
        )

        self.metrics.model_parameters = self.model_service.count_parameters(base_model)

        # Get task definition with training data
        tasks = get_predefined_tasks(self.config.fine_tuning.data_repetition_factor)
        task = tasks[self.config.task_name]

        # Initialize fine-tuner with configuration
        fine_tuner = FineTuner(
            output_dir=f"{self.config.output_dir}/finetuned_model",
            num_epochs=self.config.fine_tuning.num_epochs,
            learning_rate=self.config.fine_tuning.learning_rate,
            batch_size=self.config.fine_tuning.batch_size,
            max_length=self.config.fine_tuning.max_length,
            save_strategy=self.config.fine_tuning.save_strategy,
            logging_steps=self.config.fine_tuning.logging_steps,
            seed=self.config.seed if self.config.seed is not None else 42,
        )

        # Fine-tune the model (note: this modifies base_model in-place)
        finetuned_model, ft_metrics = fine_tuner.fine_tune(
            base_model=base_model,
            tokenizer=tokenizer,
            train_texts=task.train_texts,
        )

        # Update experiment metrics with fine-tuning results
        self.metrics.training_examples = ft_metrics["training_examples"]
        self.metrics.num_epochs = ft_metrics["num_epochs"]
        self.metrics.learning_rate = ft_metrics["learning_rate"]
        self.metrics.final_training_loss = ft_metrics["final_loss"]
        self.metrics.training_steps = ft_metrics["training_steps"]
        self.metrics.training_history = ft_metrics["training_history"]
        self.metrics.finetuning_start_time = ft_metrics["start_time"]
        self.metrics.finetuning_end_time = ft_metrics["end_time"]
        self.metrics.finetuning_duration_seconds = ft_metrics["duration_seconds"]

        # CRITICAL: Reload the base model since fine-tuning modified it in-place
        print("\nReloading base model to preserve original weights...")
        base_model_reloaded, _ = self.model_service.load_model_and_tokenizer(
            self.config.model_name, self.device
        )

        # Save models for future analysis
        print("\nSaving models for future analysis...")
        self.model_service.save_models(base_model_reloaded, finetuned_model, self.config.output_dir)
        print(f"Models saved to {self.config.output_dir}/")

        return base_model_reloaded, finetuned_model, tokenizer

    def _compute_task_vector(self, base_model, finetuned_model):
        """Compute task vector from models.

        Args:
            base_model: Base model
            finetuned_model: Fine-tuned model

        Returns:
            Task vector dictionary
        """
        print_banner("COMPUTING TASK VECTOR")

        start_time = time.time()

        task_vector = self.task_vector_service.compute(base_model, finetuned_model)
        magnitude = self.task_vector_service.compute_magnitude(task_vector)

        elapsed = time.time() - start_time

        print(f"Task vector computed: ||T|| = {magnitude:.2f} (Euclidean)")
        print(f"Computation time: {elapsed:.2f}s")

        # Update metrics
        self.metrics.task_vector_magnitude = magnitude
        self.metrics.task_vector_magnitude_euclidean = magnitude
        self.metrics.task_vector_computation_time = elapsed

        # Riemannian geometry computation if enabled
        if self.config.geometry.enabled:
            print("\nComputing Riemannian task vector metrics...")
            self._compute_riemannian_metrics(base_model, task_vector)

        print()  # Blank line
        return task_vector

    def _run_alpha_sweep(self, base_model, task_vector, tokenizer):
        """Run alpha sweep experiment.

        Args:
            base_model: Base model
            task_vector: Task vector (T1)
            tokenizer: Tokenizer

        Returns:
            Tuple of (results, analysis)
        """
        # Get task definition
        tasks = get_predefined_tasks(self.config.fine_tuning.data_repetition_factor)
        task = tasks[self.config.task_name]

        # Load general evaluation dataset with category labels
        from sitv.data.loader import DatasetLoader

        loader = DatasetLoader()
        general_eval_texts, general_eval_categories = loader.load_general_with_categories(
            self.config.evaluation.general_dataset
        )

        # For sentiment tasks, load opposite sentiment for preference calculation
        opposite_sentiment_eval_texts = None
        if "sentiment" in self.config.task_name:
            opposite_task_name = None
            if "positive" in self.config.task_name:
                opposite_task_name = "sentiment_negative_eval"
            elif "negative" in self.config.task_name:
                opposite_task_name = "sentiment_positive_eval"

            if opposite_task_name:
                try:
                    opposite_sentiment_eval_texts = loader.load_eval(opposite_task_name)
                    print(
                        f"  Loaded {len(opposite_sentiment_eval_texts)} opposite sentiment examples for preference calculation"
                    )
                except Exception as e:
                    print(f"  Warning: Could not load opposite sentiment texts: {e}")

        # Geometry context
        geometry_service = getattr(self, "geometry_service", None)
        fisher_metric = getattr(self, "fisher_metric", None)

        # Create and run experiment
        experiment = AlphaSweepExperiment(
            base_model=base_model,
            task_vector=task_vector,
            tokenizer=tokenizer,
            general_eval_texts=general_eval_texts,  # Use general dataset for L(α)
            general_eval_categories=general_eval_categories,  # Category labels for breakdown
            task_eval_texts=task.eval_texts,  # Use task-specific for task performance
            opposite_sentiment_eval_texts=opposite_sentiment_eval_texts,  # Opposite sentiment for preference
            alpha_range=self.config.alpha_sweep.alpha_range,
            num_samples=self.config.alpha_sweep.num_samples,
            device=self.device,
            enable_squaring_test=self.config.alpha_sweep.enable_squaring_test,
            sampling_strategy=self.config.alpha_sweep.sampling.strategy,
            sampling_config=self.config.alpha_sweep.sampling,
            eval_batch_size=self.config.evaluation.batch_size,
            eval_enable_mixed_precision=self.config.evaluation.enable_mixed_precision,
            eval_max_length=self.config.evaluation.max_length,
            geometry_service=geometry_service,  # Pass geometry service if available
            fisher_metric=fisher_metric,  # Pass Fisher metric if available
            geometry_config=self.config.geometry,  # Pass geometry configuration
        )

        # Run experiment
        results, timing_metadata = experiment.run()

        # Update metrics
        self.metrics.sweep_start_time = timing_metadata["start_time"]
        self.metrics.sweep_end_time = timing_metadata["end_time"]
        self.metrics.sweep_duration_seconds = timing_metadata["duration_seconds"]
        self.metrics.num_alpha_samples = self.config.alpha_sweep.num_samples
        self.metrics.alpha_range = self.config.alpha_sweep.alpha_range
        self.metrics.time_per_alpha_seconds = timing_metadata.get("time_per_alpha_seconds", 0.0)
        self.metrics.enable_squaring_test = self.config.alpha_sweep.enable_squaring_test
        self.metrics.sampling_strategy = self.config.alpha_sweep.sampling.strategy

        # Analyze results
        analyzer = ResultAnalyzer(threshold=self.config.alpha_sweep.threshold)
        analysis = analyzer.analyze(results)

        # Update metrics with analysis (only if results available)
        if analysis["min_general_loss"] is not None:
            self.metrics.min_general_loss_alpha = analysis["min_general_loss"].alpha
            self.metrics.min_general_loss = analysis["min_general_loss"].loss
        if analysis["min_task_loss"] is not None:
            self.metrics.min_task_loss_alpha = analysis["min_task_loss"].alpha
            self.metrics.min_task_loss = analysis["min_task_loss"].task_eval_loss

        self.metrics.num_zero_crossings = len(analysis["zero_crossings"])
        self.metrics.zero_crossing_alphas = [zc.alpha for zc in analysis["zero_crossings"]]

        if analysis.get("has_squaring_data", False):
            self.metrics.num_squaring_return_points = len(analysis["squaring_return_points"])
            self.metrics.squaring_return_alphas = [
                sp.alpha for sp in analysis["squaring_return_points"]
            ]

        return results, analysis

    def _run_2d_composition(self, base_model, task_vector, tokenizer):
        """Run 2D composition experiment.

        Note:
            To avoid mutating the evaluation base, this method loads a
            pristine base model for evaluation, and a separate fresh copy
            for fine-tuning the second task (T2).
        """
        print_banner("2D COMPOSITION EXPERIMENT")

        # Keep a pristine base for evaluation
        base_for_eval, _ = self.model_service.load_model_and_tokenizer(
            self.config.model_name, self.device
        )

        # Get available tasks
        tasks = get_predefined_tasks(self.config.fine_tuning.data_repetition_factor)

        # Choose second task
        if self.config.task_name == "sentiment_positive":
            second_task_name = "sentiment_negative"
        elif self.config.task_name == "sentiment_negative":
            second_task_name = "sentiment_positive"
        else:
            second_task_name = "sentiment_negative"

        if second_task_name not in tasks:
            print(f"\n⚠️  Second task '{second_task_name}' not available.")
            print(f"   Available tasks: {list(tasks.keys())}")
            print("   Skipping 2D composition experiment.")
            return

        task2 = tasks[second_task_name]

        print(f"\nFirst task:  {self.config.task_name}")
        print(f"Second task: {second_task_name}")
        print(
            f"Grid: {self.config.composition_2d.num_samples_per_dim}×{self.config.composition_2d.num_samples_per_dim} = "
            f"{self.config.composition_2d.num_samples_per_dim**2} evaluations"
        )

        # Fine-tune on second task to create T2 using a fresh copy (do NOT mutate base_for_eval)
        print_banner(f"FINE-TUNING ON SECOND TASK: {second_task_name}")
        base_for_ft, _ = self.model_service.load_model_and_tokenizer(
            self.config.model_name, self.device
        )
        fine_tuner = FineTuner(
            output_dir=f"{self.config.output_dir}/finetuned_model_2",
            num_epochs=self.config.fine_tuning.num_epochs,
            learning_rate=self.config.fine_tuning.learning_rate,
            batch_size=self.config.fine_tuning.batch_size,
            max_length=self.config.fine_tuning.max_length,
            save_strategy=self.config.fine_tuning.save_strategy,
            logging_steps=self.config.fine_tuning.logging_steps,
            seed=self.config.seed if self.config.seed is not None else 42,
        )
        finetuned_model_2, _ = fine_tuner.fine_tune(
            base_model=base_for_ft,
            tokenizer=tokenizer,
            train_texts=task2.train_texts,
        )

        # Compute second task vector against a fresh base
        print_banner("COMPUTING SECOND TASK VECTOR")
        start_time = time.time()
        base_model_for_task2, _ = self.model_service.load_model_and_tokenizer(
            self.config.model_name, self.device
        )
        task_vector_2 = self.task_vector_service.compute(base_model_for_task2, finetuned_model_2)
        magnitude_2 = self.task_vector_service.compute_magnitude(task_vector_2)
        elapsed = time.time() - start_time

        print(f"Task vector 2 computed: ||T2|| = {magnitude_2:.2f}")
        print(f"Computation time: {elapsed:.2f}s\n")

        # Store second task vector magnitude and name in metrics
        self.metrics.enable_2d_composition = True
        self.metrics.task_name_2 = second_task_name
        self.metrics.task_vector_2_magnitude = magnitude_2

        # Load general evaluation dataset (same as 1D alpha sweep)
        from sitv.data.loader import DatasetLoader

        loader = DatasetLoader()
        general_eval_texts = loader.load_general(self.config.evaluation.general_dataset)

        # Run 2D composition experiment on a pristine base model
        experiment = Composition2DExperiment(
            base_model=base_for_eval,  # pristine!
            task_vector_1=task_vector,
            task_vector_2=task_vector_2,
            tokenizer=tokenizer,
            general_eval_texts=general_eval_texts,  # L(α,β)
            alpha_range=self.config.composition_2d.alpha_range,
            beta_range=self.config.composition_2d.beta_range,
            num_samples_per_dim=self.config.composition_2d.num_samples_per_dim,
            device=self.device,
            eval_batch_size=self.config.evaluation.batch_size,
            eval_enable_mixed_precision=self.config.evaluation.enable_mixed_precision,
            eval_max_length=self.config.evaluation.max_length,
        )

        results_2d, _ = experiment.run()

        # Store results for report generation
        self.results_2d = results_2d

        # Generate and save 2D plot
        print_banner("GENERATING 2D VISUALIZATION")

        plotter = ResultPlotter()
        plot_path_2d = f"{self.config.output_dir}/loss_landscape_2d.png"
        plotter.plot_2d_composition(results_2d, plot_path_2d)

        # Save 2D results
        results_2d_path = f"{self.config.output_dir}/loss_landscape_2d_results.json"
        with open(results_2d_path, "w") as f:
            json.dump(
                [
                    {
                        "alpha": r.alpha,
                        "beta": r.beta,
                        "loss": r.loss,
                        "base_loss": r.base_loss,
                        "functional_return": r.functional_return,
                        "perplexity": r.perplexity,
                    }
                    for r in results_2d
                ],
                f,
                indent=2,
            )

        print(f"2D results saved: {results_2d_path}")

        # Run composition analysis if enabled
        self._run_composition_analysis()

        print_banner("2D COMPOSITION EXPERIMENT COMPLETE")
        print_separator()
        print()

    def _run_3d_composition(self, base_model, task_vector, tokenizer):
        """Run 3D composition experiment.

        Note:
            Uses a pristine base for evaluation, with separate fresh copies
            for fine-tuning T2 and T3 to avoid mutating the evaluation base.
        """
        print_banner("3D COMPOSITION EXPERIMENT")

        # Keep a pristine base for evaluation
        base_for_eval, _ = self.model_service.load_model_and_tokenizer(
            self.config.model_name, self.device
        )

        # Get available tasks
        tasks = get_predefined_tasks(self.config.fine_tuning.data_repetition_factor)

        # Get task names from config
        task1_name = self.config.composition_3d.task_1
        task2_name = self.config.composition_3d.task_2
        task3_name = self.config.composition_3d.task_3

        # Validate tasks exist
        for task_name in [task1_name, task2_name, task3_name]:
            if task_name not in tasks:
                print(f"\n⚠️  Task '{task_name}' not available.")
                print(f"   Available tasks: {list(tasks.keys())}")
                print("   Skipping 3D composition experiment.")
                return

        print(f"\nFirst task:  {task1_name}")
        print(f"Second task: {task2_name}")
        print(f"Third task:  {task3_name}")
        print(
            f"Grid: {self.config.composition_3d.num_samples_per_dim}³ = "
            f"{self.config.composition_3d.num_samples_per_dim**3} evaluations"
        )

        # Determine which task vectors to compute
        task_vectors: dict[str, dict[str, torch.Tensor]] = {}
        task_magnitudes: dict[str, float] = {}

        # T1: reuse if same as main task
        if task1_name == self.config.task_name:
            task_vectors[task1_name] = task_vector
            task_magnitudes[task1_name] = self.task_vector_service.compute_magnitude(task_vector)
            print(
                f"\nTask 1 ('{task1_name}'): Reusing main task vector (||T1|| = {task_magnitudes[task1_name]:.2f})"
            )
        else:
            print_banner(f"FINE-TUNING ON TASK 1: {task1_name}")
            tv1, mag1 = self._fine_tune_and_compute_task_vector(
                base_model, tokenizer, tasks[task1_name], task1_name, "finetuned_model_3d_t1"
            )
            task_vectors[task1_name] = tv1
            task_magnitudes[task1_name] = mag1

        # T2
        print_banner(f"FINE-TUNING ON TASK 2: {task2_name}")
        tv2, mag2 = self._fine_tune_and_compute_task_vector(
            base_model, tokenizer, tasks[task2_name], task2_name, "finetuned_model_3d_t2"
        )
        task_vectors[task2_name] = tv2
        task_magnitudes[task2_name] = mag2

        # T3
        print_banner(f"FINE-TUNING ON TASK 3: {task3_name}")
        tv3, mag3 = self._fine_tune_and_compute_task_vector(
            base_model, tokenizer, tasks[task3_name], task3_name, "finetuned_model_3d_t3"
        )
        task_vectors[task3_name] = tv3
        task_magnitudes[task3_name] = mag3

        # Store metadata
        self.metrics.enable_3d_composition = True
        self.metrics.task_name_3d_1 = task1_name
        self.metrics.task_name_3d_2 = task2_name
        self.metrics.task_name_3d_3 = task3_name
        self.metrics.task_vector_3d_1_magnitude = task_magnitudes[task1_name]
        self.metrics.task_vector_3d_2_magnitude = task_magnitudes[task2_name]
        self.metrics.task_vector_3d_3_magnitude = task_magnitudes[task3_name]

        # Load general evaluation dataset
        from sitv.data.loader import DatasetLoader

        loader = DatasetLoader()
        general_eval_texts = loader.load_general(self.config.evaluation.general_dataset)

        # Run 3D composition experiment on a pristine base model
        print_banner("RUNNING 3D COMPOSITION SWEEP")
        experiment = Composition3DExperiment(
            base_model=base_for_eval,  # pristine!
            task_vector_1=task_vectors[task1_name],
            task_vector_2=task_vectors[task2_name],
            task_vector_3=task_vectors[task3_name],
            tokenizer=tokenizer,
            general_eval_texts=general_eval_texts,
            alpha_range=self.config.composition_3d.alpha_range,
            beta_range=self.config.composition_3d.beta_range,
            gamma_range=self.config.composition_3d.gamma_range,
            num_samples_per_dim=self.config.composition_3d.num_samples_per_dim,
            device=self.device,
            eval_batch_size=self.config.evaluation.batch_size,
            eval_enable_mixed_precision=self.config.evaluation.enable_mixed_precision,
            eval_max_length=self.config.evaluation.max_length,
        )

        results_3d, _ = experiment.run()

        # Store results for report generation
        self.results_3d = results_3d

        # Generate and save 3D visualization
        print_banner("GENERATING 3D VISUALIZATION")

        plotter = ResultPlotter()
        plot_path_interactive = f"{self.config.output_dir}/loss_landscape_3d_interactive.html"
        plot_path_slices = f"{self.config.output_dir}/loss_landscape_3d_slices.png"
        plotter.plot_3d_composition(results_3d, plot_path_interactive, plot_path_slices)

        # Save 3D results
        results_3d_path = f"{self.config.output_dir}/loss_landscape_3d_results.json"
        with open(results_3d_path, "w") as f:
            json.dump(
                [
                    {
                        "alpha": r.alpha,
                        "beta": r.beta,
                        "gamma": r.gamma,
                        "loss": r.loss,
                        "base_loss": r.base_loss,
                        "functional_return": r.functional_return,
                        "perplexity": r.perplexity,
                    }
                    for r in results_3d
                ],
                f,
                indent=2,
            )

        print(f"3D results saved: {results_3d_path}")

        print_banner("3D COMPOSITION EXPERIMENT COMPLETE")
        print_separator()
        print()

    def _fine_tune_and_compute_task_vector(
        self, base_model, tokenizer, task, task_name, output_subdir
    ):
        """Fine-tune on a task and compute its task vector safely.

        Important:
            This method intentionally loads a fresh base model for fine-tuning
            to avoid mutating any caller-provided model object.
        """
        # Fine-tune using a fresh copy
        base_for_ft, _ = self.model_service.load_model_and_tokenizer(
            self.config.model_name, self.device
        )
        fine_tuner = FineTuner(
            output_dir=f"{self.config.output_dir}/{output_subdir}",
            num_epochs=self.config.fine_tuning.num_epochs,
            learning_rate=self.config.fine_tuning.learning_rate,
            batch_size=self.config.fine_tuning.batch_size,
            max_length=self.config.fine_tuning.max_length,
            save_strategy=self.config.fine_tuning.save_strategy,
            logging_steps=self.config.fine_tuning.logging_steps,
            seed=self.config.seed if self.config.seed is not None else 42,
        )
        finetuned_model, _ = fine_tuner.fine_tune(
            base_model=base_for_ft,
            tokenizer=tokenizer,
            train_texts=task.train_texts,
        )

        # Compute task vector against a fresh base
        print(f"\nReloading base model for '{task_name}' task vector computation...")
        base_model_fresh, _ = self.model_service.load_model_and_tokenizer(
            self.config.model_name, self.device
        )

        print(f"\nComputing task vector for '{task_name}'...")
        start_time = time.time()
        task_vec = self.task_vector_service.compute(base_model_fresh, finetuned_model)
        magnitude = self.task_vector_service.compute_magnitude(task_vec)
        elapsed = time.time() - start_time

        print(f"Task vector '{task_name}' computed: ||T|| = {magnitude:.2f}")
        print(f"Computation time: {elapsed:.2f}s\n")

        return task_vec, magnitude

    def _run_composition_analysis(self):
        """Run composition analysis on 2D results."""
        if not self.config.composition_2d.enable_analysis:
            return
        if self.results_2d is None or len(self.results_2d) == 0:
            print("⚠️  Skipping composition analysis: No 2D results available")
            return

        print_banner("ANALYZING 2D COMPOSITION")
        print("Running composition analyzer to detect interaction patterns...")

        try:
            from sitv.analysis import CompositionAnalyzer

            analyzer = CompositionAnalyzer(Path(self.config.output_dir))
            self.composition_analysis = analyzer.run_full_analysis(
                save_results=True,
                save_visualization=True,
            )

            print("✓ Composition analysis complete")
            print(
                f"  - Interaction type: {self.composition_analysis.get('interaction_type', 'unknown')}"
            )
            print(f"  - R² score: {self.composition_analysis.get('r2_score', 0.0):.4f}")

        except Exception as e:
            print(f"⚠️  Composition analysis failed: {e}")
            import traceback

            traceback.print_exc()
            self.composition_analysis = None

        print_separator()
        print()

    def _generate_outputs(self, results, analysis):
        """Generate all output files.

        Args:
            results: Experiment results
            analysis: Analysis dictionary
        """
        print_banner("GENERATING OUTPUTS")

        # Save results
        self.file_manager.save_results(results)

        # Save analysis
        self.file_manager.save_analysis(analysis)

        # Generate plot
        plotter = ResultPlotter()
        plot_path = self.path_manager.get_plot_path()
        plotter.plot_alpha_sweep(
            results,
            analysis,
            plot_path,
            self.config.alpha_sweep.enable_squaring_test,
        )

        # Generate markdown report
        report_generator = MarkdownReportGenerator()
        report_path = self.path_manager.get_report_path()
        report_generator.generate(
            results,
            analysis,
            self.metrics,
            report_path,
            results_2d=self.results_2d,
            results_3d=self.results_3d,
            composition_analysis=self.composition_analysis,
            curvature_results=getattr(self, "curvature_results", None),
            symmetry_results=getattr(self, "symmetry_results", None),
        )

        # Save metrics
        self.file_manager.save_metrics(self.metrics)

    def _compute_riemannian_metrics(self, base_model, task_vector):
        """Compute Riemannian geometry metrics if enabled.

        Args:
            base_model: Base model
            task_vector: Task vector dictionary
        """
        from sitv.geometry import GeodesicTaskVectorService

        # Get training data for Fisher computation
        tasks = get_predefined_tasks(self.config.fine_tuning.data_repetition_factor)
        task = tasks[self.config.task_name]
        training_texts = task.train_texts[: self.config.geometry.fisher_approximation.num_samples]

        # Initialize geometry service
        geo_service = GeodesicTaskVectorService(
            config=self.config.geometry,
            tokenizer=self.model_service.load_tokenizer(self.config.model_name),
            device=self.device,
        )

        # Compute Fisher metric
        start_fisher = time.time()
        fisher = geo_service.get_or_compute_fisher(
            base_model,
            training_texts,
            cache_key="base",
        )
        fisher_time = time.time() - start_fisher

        # Compute Riemannian norm
        from typing import cast

        riem_magnitude = geo_service.compute_magnitude(
            task_vector, cast(dict[str, torch.Tensor] | None, fisher)
        )

        # Compute Christoffel symbols and detect curvature (if requested)
        christoffel_rms = 0.0
        curvature_detected = False
        if self.config.geometry.geodesic_integration.recompute_metric_every > 0:
            christoffel_start = time.time()

            # Get num_samples from christoffel_computation config
            num_samples = self.config.geometry.christoffel_computation.num_samples
            num_samples = min(num_samples, len(training_texts))

            # Compute Christoffel symbols via finite differences using CLONED params
            christoffel = geo_service.fisher_service.compute_christoffel_symbols_finite_diff(
                model=base_model,
                base_params={
                    name: param.data.clone() for name, param in base_model.named_parameters()
                },
                data_texts=training_texts[:num_samples],
                epsilon=self.config.geometry.geodesic_integration.metric_epsilon,
                config=self.config.geometry.christoffel_computation,
            )

            # Compute RMS of Christoffel symbols
            christoffel_squared_sum = sum(
                torch.sum(gamma**2).item() for gamma in christoffel.values()
            )
            total_params = sum(gamma.numel() for gamma in christoffel.values())
            christoffel_rms = (
                (christoffel_squared_sum / total_params) ** 0.5 if total_params > 0 else 0.0
            )

            # Detect curvature (threshold: RMS > 1e-4 indicates non-trivial curvature)
            curvature_detected = christoffel_rms > 1e-4

            christoffel_time = time.time() - christoffel_start
            print(f"  Christoffel symbols computed in {christoffel_time:.2f}s")
            print(f"  RMS(Γ) = {christoffel_rms:.6f}")
            print(f"  Curvature detected: {'✅ YES' if curvature_detected else '❌ NO'}")

        # Store in metrics
        self.metrics.geometry_enabled = True
        self.metrics.metric_type = self.config.geometry.metric_type.value
        self.metrics.fisher_computation_time = fisher_time
        self.metrics.fisher_num_samples = len(training_texts)
        self.metrics.task_vector_magnitude_riemannian = riem_magnitude
        self.metrics.geodesic_integration_enabled = (
            self.config.geometry.geodesic_integration.enabled
        )
        self.metrics.geodesic_num_steps = self.config.geometry.geodesic_integration.num_steps
        self.metrics.christoffel_rms = christoffel_rms
        self.metrics.curvature_detected = curvature_detected
        self.metrics.recompute_metric_every = (
            self.config.geometry.geodesic_integration.recompute_metric_every
        )
        self.metrics.metric_recompute_count = (
            (self.metrics.geodesic_num_steps // self.metrics.recompute_metric_every)
            if self.metrics.recompute_metric_every > 0
            else 0
        )

        # Store geometry service for later use (alpha sweep, etc.)
        self.geometry_service = geo_service

        # Clear model gradients from Fisher computation to free GPU memory
        base_model.zero_grad(set_to_none=True)

        # Move Fisher metric to CPU to free GPU memory for alpha sweep
        self.fisher_metric = {
            name: (tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor)
            for name, tensor in fisher.items()
        }

        # Delete any remaining GPU references
        del fisher

        # Clear accelerator caches to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            try:
                # PyTorch MPS cache clear (available in recent builds)
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    # noinspection PyUnresolvedReferences
                    torch.mps.empty_cache()  # type: ignore[attr-defined]
            except Exception:
                pass

        den = self.metrics.task_vector_magnitude_euclidean or 0.0
        ratio = riem_magnitude / (den if den > 1e-12 else 1e-12)
        print(f"  Fisher metric computed in {fisher_time:.2f}s")
        print(f"  Riemannian magnitude: ||T||_g = {riem_magnitude:.4f}")
        print(f"  Ratio ||T||_g / ||T|| = {ratio:.4f}")

        # Optional: curvature analysis
        if self.config.geometry.curvature_analysis.enabled:
            print("\n  Computing curvature analysis...")
            self._compute_curvature_analysis(base_model, self.fisher_metric)

        # Optional: symmetry analysis
        if self.config.geometry.symmetry_analysis.enabled:
            print("\n  Computing symmetry analysis...")
            self._compute_symmetry_analysis(base_model)

    def _compute_curvature_analysis(self, base_model, fisher):
        """Compute curvature analysis on parameter manifold."""
        from sitv.geometry import CurvatureAnalyzer

        # Get base parameters (cloned)
        base_params = {name: param.data.clone() for name, param in base_model.named_parameters()}

        # Initialize curvature analyzer
        curvature_analyzer = CurvatureAnalyzer(
            config=self.config.geometry.curvature_analysis, device=self.device
        )

        # Compute curvature distribution
        start_time = time.time()
        curvature_results = curvature_analyzer.estimate_curvature_distribution(
            base_point=base_params,
            fisher=fisher,
            num_samples=self.config.geometry.curvature_analysis.num_tangent_samples,
        )
        curvature_time = time.time() - start_time

        # Store results for later use in reporting
        self.curvature_results = curvature_results

        # Print summary
        print(f"    Mean curvature: {curvature_results['mean_curvature']:.6f}")
        print(
            f"    Curvature range: [{curvature_results['min_curvature']:.6f}, "
            f"{curvature_results['max_curvature']:.6f}]"
        )
        print(f"    Interpretation: {curvature_results['interpretation']}")
        print(f"    Computed in {curvature_time:.2f}s ({curvature_results['num_samples']} samples)")

    def _compute_symmetry_analysis(self, base_model):
        """Compute symmetry analysis on parameter space."""
        from sitv.core.evaluation import EvaluationService
        from sitv.geometry import SymmetryAnalyzer

        # Get evaluation texts
        tasks = get_predefined_tasks(self.config.fine_tuning.data_repetition_factor)
        task = tasks[self.config.task_name]
        eval_texts = task.eval_texts[:100]  # Use subset for faster symmetry testing

        # Initialize evaluator for symmetry testing
        tokenizer = self.model_service.load_tokenizer(self.config.model_name)
        evaluator = EvaluationService(
            tokenizer=tokenizer,
            device=self.device,
            batch_size=self.config.evaluation.batch_size,
            max_length=self.config.evaluation.max_length,
            enable_mixed_precision=self.config.evaluation.enable_mixed_precision,
        )

        # Initialize symmetry analyzer
        symmetry_analyzer = SymmetryAnalyzer(
            config=self.config.geometry.symmetry_analysis, evaluator=evaluator, device=self.device
        )

        # Analyze all symmetries
        start_time = time.time()
        symmetry_results = symmetry_analyzer.analyze_all_symmetries(
            model=base_model,
            eval_texts=eval_texts,
            num_tests_per_type=10,
        )
        symmetry_time = time.time() - start_time

        # Store results for later use in reporting
        self.symmetry_results = symmetry_results

        # Print summary
        print(f"    Symmetries detected: {symmetry_results['summary']['num_symmetries_detected']}")
        if "rotation" in symmetry_results:
            print(f"    Rotation symmetry: {symmetry_results['rotation']['symmetry_score']:.2f}")
        if "permutation" in symmetry_results:
            print(
                f"    Permutation symmetry: {symmetry_results['permutation']['symmetry_score']:.2f}"
            )
        if "scaling" in symmetry_results:
            print(f"    Scaling symmetry: {symmetry_results['scaling']['symmetry_score']:.2f}")
        print(f"    Computed in {symmetry_time:.2f}s")

        # Project to quotient space if enabled
        if self.config.geometry.symmetry_analysis.quotient_space:
            detected_symmetries = [
                sym_type
                for sym_type in ["rotation", "permutation", "scaling"]
                if symmetry_results.get(sym_type, {}).get("is_symmetric", False)
            ]
            if detected_symmetries:
                print(f"    Projecting to quotient space (modulo {', '.join(detected_symmetries)})")
                symmetry_analyzer.project_to_quotient_space(base_model, detected_symmetries)

    def _finalize_metrics(self):
        """Finalize experiment metrics."""
        self.metrics.end_time = datetime.now().isoformat()
        experiment_end = time.time()
        experiment_start = datetime.fromisoformat(self.metrics.start_time).timestamp()
        self.metrics.duration_seconds = experiment_end - experiment_start

        # Store geometry configuration if not already set
        if not getattr(self.metrics, "geometry_enabled", False) and self.config.geometry.enabled:
            # Geometry was enabled in config but no Riemannian metrics were computed.
            self.metrics.geometry_enabled = False
            self.metrics.metric_type = self.config.geometry.metric_type.value
