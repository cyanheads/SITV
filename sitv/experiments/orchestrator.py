"""
Experiment orchestrator for SITV.

This module provides the ExperimentOrchestrator that coordinates the entire
experimental workflow from model loading to result generation.
"""

import time
from datetime import datetime
from typing import Optional

from sitv.experiments.config import ExperimentConfig
from sitv.data.models import ExperimentMetrics
from sitv.data.tasks import get_predefined_tasks
from sitv.models import ModelService
from sitv.core import TaskVectorService, get_device, get_device_string
from sitv.experiments import AlphaSweepExperiment, Composition2DExperiment
from sitv.analysis import ResultAnalyzer
from sitv.reporting import MarkdownReportGenerator
from sitv.visualization import ResultPlotter
from sitv.io import FileManager, PathManager


class ExperimentOrchestrator:
    """Orchestrator for SITV experiments.

    This orchestrator coordinates the entire experiment workflow:
    1. Load or fine-tune models
    2. Compute task vectors
    3. Run experiments (alpha sweep, 2D composition)
    4. Analyze results
    5. Generate reports and visualizations
    6. Save all outputs

    Attributes:
        config: Experiment configuration
        metrics: Experiment metrics tracker
        model_service: Model management service
        task_vector_service: Task vector computation service
        file_manager: File I/O manager
        path_manager: Path management
        device: Computation device
    """

    def __init__(self, config: ExperimentConfig):
        """Initialize the experiment orchestrator.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.metrics = ExperimentMetrics(
            start_time=datetime.now().isoformat()
        )

        # Initialize services
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

    def run(self) -> None:
        """Run the complete experiment workflow.

        This is the main entry point that executes all experiment phases.
        """
        print(f"\n{'='*70}")
        print("SITV EXPERIMENT START")
        print(f"{'='*70}")
        print(f"Model: {self.config.model_name}")
        print(f"Task: {self.config.task_name}")
        print(f"Device: {self.device}")
        print(f"Output: {self.config.output_dir}")
        print(f"Analysis Only: {self.config.analysis_only}")
        print(f"{'='*70}\n")

        # Phase 1: Load or fine-tune models
        if self.config.analysis_only:
            base_model, finetuned_model, tokenizer = self._load_saved_models()
        else:
            base_model, finetuned_model, tokenizer = self._fine_tune_and_save()

        # Phase 2: Compute task vector
        task_vector = self._compute_task_vector(base_model, finetuned_model)

        # Phase 3: Run alpha sweep experiment
        results, analysis = self._run_alpha_sweep(
            base_model,
            task_vector,
            tokenizer
        )

        # Phase 4: Run 2D composition (if enabled)
        if self.config.enable_2d_composition:
            self._run_2d_composition(base_model, task_vector, tokenizer)

        # Phase 5: Generate outputs
        self._generate_outputs(results, analysis)

        # Finalize metrics
        self._finalize_metrics()

        print(f"\n{'='*70}")
        print("SITV EXPERIMENT COMPLETE")
        print(f"{'='*70}")
        print(f"Duration: {self.metrics.duration_seconds / 60:.1f} minutes")
        print(f"Output directory: {self.config.output_dir}")
        print(f"{'='*70}\n")

    def _load_saved_models(self):
        """Load previously saved models for analysis.

        Returns:
            Tuple of (base_model, finetuned_model, tokenizer)
        """
        print("\n" + "="*70)
        print("LOADING SAVED MODELS")
        print("="*70)

        # Check if models exist
        if not self.model_service.check_saved_models_exist(self.config.output_dir):
            raise FileNotFoundError(
                f"Saved models not found in {self.config.output_dir}. "
                "Run without --analysis-only first to fine-tune models."
            )

        # Load models
        base_model, finetuned_model = self.model_service.load_saved_models(
            self.config.output_dir,
            self.device
        )

        # Load tokenizer
        tokenizer = self.model_service.load_tokenizer(self.config.model_name)

        # Update metrics
        self.metrics.model_parameters = self.model_service.count_parameters(base_model)

        return base_model, finetuned_model, tokenizer

    def _fine_tune_and_save(self):
        """Fine-tune model and save for future analysis.

        Returns:
            Tuple of (base_model, finetuned_model, tokenizer)

        Note:
            This is a placeholder. The actual fine-tuning logic from main.py
            should be migrated here or to sitv.models.fine_tuner.
        """
        print("\n" + "="*70)
        print("MODEL FINE-TUNING")
        print("="*70)

        # Load base model and tokenizer
        base_model, tokenizer = self.model_service.load_model_and_tokenizer(
            self.config.model_name,
            self.device
        )

        self.metrics.model_parameters = self.model_service.count_parameters(base_model)

        # TODO: Implement fine-tuning using FineTuner service
        # For now, this is a placeholder that indicates where the
        # fine-tuning logic from main.py should be integrated

        print("\nNOTE: Fine-tuning logic needs to be migrated from main.py")
        print("For now, use --analysis-only mode with pre-trained models")

        raise NotImplementedError(
            "Fine-tuning integration pending. "
            "Please use --analysis-only mode or integrate fine-tuning logic."
        )

    def _compute_task_vector(self, base_model, finetuned_model):
        """Compute task vector from models.

        Args:
            base_model: Base model
            finetuned_model: Fine-tuned model

        Returns:
            Task vector dictionary
        """
        print("\n" + "="*70)
        print("COMPUTING TASK VECTOR")
        print("="*70)

        start_time = time.time()

        task_vector = self.task_vector_service.compute(base_model, finetuned_model)
        magnitude = self.task_vector_service.compute_magnitude(task_vector)

        elapsed = time.time() - start_time

        print(f"Task vector computed: ||T|| = {magnitude:.2f}")
        print(f"Computation time: {elapsed:.2f}s\n")

        # Update metrics
        self.metrics.task_vector_magnitude = magnitude
        self.metrics.task_vector_computation_time = elapsed

        return task_vector

    def _run_alpha_sweep(self, base_model, task_vector, tokenizer):
        """Run alpha sweep experiment.

        Args:
            base_model: Base model
            task_vector: Task vector
            tokenizer: Tokenizer

        Returns:
            Tuple of (results, analysis)
        """
        # Get task definition
        tasks = get_predefined_tasks()
        task = tasks[self.config.task_name]

        # Create experiment
        experiment = AlphaSweepExperiment(
            base_model=base_model,
            task_vector=task_vector,
            tokenizer=tokenizer,
            general_eval_texts=task.eval_texts,
            task_eval_texts=task.eval_texts,
            alpha_range=self.config.alpha_sweep.alpha_range,
            num_samples=self.config.alpha_sweep.num_samples,
            device=self.device,
            enable_squaring_test=self.config.alpha_sweep.enable_squaring_test,
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

        # Analyze results
        analyzer = ResultAnalyzer(threshold=self.config.alpha_sweep.threshold)
        analysis = analyzer.analyze(results)

        # Update metrics with analysis
        self.metrics.min_general_loss_alpha = analysis["min_general_loss"].alpha
        self.metrics.min_general_loss = analysis["min_general_loss"].loss
        self.metrics.min_task_loss_alpha = analysis["min_task_loss"].alpha
        self.metrics.min_task_loss = analysis["min_task_loss"].task_performance
        self.metrics.num_zero_crossings = len(analysis["zero_crossings"])
        self.metrics.zero_crossing_alphas = [zc.alpha for zc in analysis["zero_crossings"]]

        if analysis["has_squaring_data"]:
            self.metrics.num_squaring_return_points = len(analysis["squaring_return_points"])
            self.metrics.squaring_return_alphas = [
                sp.alpha for sp in analysis["squaring_return_points"]
            ]

        return results, analysis

    def _run_2d_composition(self, base_model, task_vector, tokenizer):
        """Run 2D composition experiment.

        Args:
            base_model: Base model
            task_vector: First task vector
            tokenizer: Tokenizer

        Note:
            This requires a second task vector. Implementation pending.
        """
        print("\n2D composition experiment enabled but not yet fully integrated")
        print("TODO: Implement second task vector creation and composition experiment")

    def _generate_outputs(self, results, analysis):
        """Generate all output files.

        Args:
            results: Experiment results
            analysis: Analysis dictionary
        """
        print("\n" + "="*70)
        print("GENERATING OUTPUTS")
        print("="*70)

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
            self.config.alpha_sweep.enable_squaring_test
        )

        # Generate markdown report
        report_generator = MarkdownReportGenerator()
        report_path = self.path_manager.get_report_path()
        report_generator.generate(results, analysis, self.metrics, report_path)

        # Save metrics
        self.file_manager.save_metrics(self.metrics)

        print(f"\n{'='*70}")

    def _finalize_metrics(self):
        """Finalize experiment metrics."""
        self.metrics.end_time = datetime.now().isoformat()
        experiment_end = time.time()
        experiment_start = datetime.fromisoformat(self.metrics.start_time).timestamp()
        self.metrics.duration_seconds = experiment_end - experiment_start
