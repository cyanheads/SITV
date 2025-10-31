"""
Markdown report generation for SITV experiments.

This module provides the MarkdownReportGenerator for creating comprehensive
experiment reports in Markdown format for LLM analysis.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from sitv.data.models import AlphaSweepResult, ExperimentMetrics, TwoDSweepResult


class MarkdownReportGenerator:
    """Service for generating Markdown experiment reports.

    This generator creates comprehensive reports for LLM analysis including:
    - Executive summary
    - Configuration details
    - Timing breakdown
    - Loss landscape analysis
    - Statistical summaries
    - Practical recommendations
    - Connection to theoretical background

    NOTE: This is a refactored version. The original 328-line generate_markdown_report()
    function from main.py should be fully migrated here. For now, this provides the
    essential structure.
    """

    def __init__(self):
        """Initialize the markdown report generator."""
        pass

    def generate(
        self,
        results: List[AlphaSweepResult],
        analysis: Dict[str, Any],
        metrics: ExperimentMetrics,
        output_path: str = "experiment_report.md",
        results_2d: Optional[List[TwoDSweepResult]] = None
    ) -> str:
        """Generate comprehensive Markdown report.

        Args:
            results: List of alpha sweep results
            analysis: Analysis dictionary from ResultAnalyzer
            metrics: ExperimentMetrics object with full experiment data
            output_path: Path to save report
            results_2d: Optional list of 2D composition results

        Returns:
            Path to saved report

        Examples:
            >>> generator = MarkdownReportGenerator()
            >>> report_path = generator.generate(results, analysis, metrics)
        """
        report = self._build_report(results, analysis, metrics, results_2d)

        # Write report
        with open(output_path, 'w') as f:
            f.write(report)

        print(f"\nMarkdown report saved: {output_path}")
        return output_path

    def _build_report(
        self,
        results: List[AlphaSweepResult],
        analysis: Dict[str, Any],
        metrics: ExperimentMetrics,
        results_2d: Optional[List[TwoDSweepResult]] = None
    ) -> str:
        """Build the complete report content.

        Args:
            results: List of results
            analysis: Analysis dictionary
            metrics: Experiment metrics
            results_2d: Optional list of 2D composition results

        Returns:
            Complete Markdown report as string
        """
        sections = []

        sections.append(self._create_header(metrics))
        sections.append(self._create_executive_summary(analysis, metrics))
        sections.append(self._create_configuration_section(metrics))
        sections.append(self._create_timing_breakdown(metrics))
        sections.append(self._create_training_history(metrics))
        sections.append(self._create_results_summary(analysis))
        sections.append(self._create_alpha_sweep_details(results, metrics))
        sections.append(self._create_statistical_summary(results, analysis))

        # Add category breakdown if available
        if results and results[0].category_losses:
            sections.append(self._create_category_breakdown(results, metrics))

        # Add squaring test analysis if available
        if analysis.get("has_squaring_data", False):
            sections.append(self._create_squaring_test_analysis(analysis))

        # Add 2D composition analysis if available
        if results_2d is not None and len(results_2d) > 0:
            sections.append(self._create_2d_composition_section(results_2d, metrics))

        sections.append(self._create_theoretical_connection(analysis))
        sections.append(self._create_recommendations(analysis))

        return "\n\n".join(sections)

    def _create_header(self, metrics: ExperimentMetrics) -> str:
        """Create report header."""
        return f"""# SITV Experiment Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model**: {metrics.model_name}
**Task**: {metrics.task_name}
**General Evaluation Dataset**: {metrics.general_eval_dataset}
**Device**: {metrics.device}"""

    def _create_executive_summary(
        self,
        analysis: Dict[str, Any],
        metrics: ExperimentMetrics
    ) -> str:
        """Create executive summary section."""
        min_general = analysis["min_general_loss"]
        min_task = analysis["min_task_loss"]
        zero_crossings = len(analysis["zero_crossings"])

        # Safe duration formatting
        total_duration = metrics.duration_seconds / 60 if metrics.duration_seconds is not None else 0

        return f"""## Executive Summary

- **Best General α**: {min_general.alpha:+.4f} (Loss: {min_general.loss:.4f})
- **Best Task α**: {min_task.alpha:+.4f} (Loss: {min_task.task_performance:.4f})
- **Zero-Crossings**: {zero_crossings} found
- **Total Duration**: {total_duration:.1f} minutes"""

    def _create_configuration_section(self, metrics: ExperimentMetrics) -> str:
        """Create configuration section."""
        return f"""## Configuration

### Model
- **Model Name**: {metrics.model_name}
- **Total Parameters**: {metrics.model_parameters:,}
- **Device**: {metrics.device}

### Task & Evaluation
- **Training Task**: {metrics.task_name}
- **General Evaluation Dataset**: {metrics.general_eval_dataset}
  - *This dataset measures how the task vector affects general language modeling capability*

### Training
- **Training Examples**: {metrics.training_examples}
- **Epochs**: {metrics.num_epochs}
- **Learning Rate**: {metrics.learning_rate:.2e}
- **Training Steps**: {metrics.training_steps}
- **Final Training Loss**: {metrics.final_training_loss:.4f}

### Task Vector
- **Magnitude (||T||)**: {metrics.task_vector_magnitude:.4f}
- **Computation Time**: {metrics.task_vector_computation_time:.2f}s

### Alpha Sweep
- **Alpha Range**: [{metrics.alpha_range[0]}, {metrics.alpha_range[1]}]
- **Samples**: {metrics.num_alpha_samples}
- **Sampling Strategy**: {metrics.sampling_strategy.capitalize()}
- **Avg Time per Sample**: {metrics.time_per_alpha_seconds:.2f}s"""

    def _create_timing_breakdown(self, metrics: ExperimentMetrics) -> str:
        """Create timing breakdown section."""
        # Guard against division by zero
        ft_pct = (metrics.finetuning_duration_seconds / metrics.duration_seconds * 100) if metrics.duration_seconds > 0 else 0
        sweep_pct = (metrics.sweep_duration_seconds / metrics.duration_seconds * 100) if metrics.duration_seconds > 0 else 0

        # Safe duration formatting
        ft_duration = metrics.finetuning_duration_seconds / 60 if metrics.finetuning_duration_seconds is not None else 0
        sweep_duration = metrics.sweep_duration_seconds / 60 if metrics.sweep_duration_seconds is not None else 0
        total_duration = metrics.duration_seconds / 60 if metrics.duration_seconds is not None else 0

        return f"""## Timing Breakdown

| Phase | Duration | Percentage |
|-------|----------|------------|
| Fine-tuning | {ft_duration:.1f}m | {ft_pct:.1f}% |
| Alpha Sweep | {sweep_duration:.1f}m | {sweep_pct:.1f}% |
| **Total** | **{total_duration:.1f}m** | **100%** |"""

    def _create_training_history(self, metrics: ExperimentMetrics) -> str:
        """Create training history section with step-by-step metrics.

        Args:
            metrics: Experiment metrics with training history

        Returns:
            Training history section as string
        """
        if not metrics.training_history:
            return """## Training History

**No training history available.**"""

        # Get key training milestones (every 5 steps or at epoch boundaries)
        milestones = []
        for i, entry in enumerate(metrics.training_history):
            step = entry.get('step', i + 1)
            # Include every 5th step or steps near epoch boundaries
            if step % 5 == 0 or i == 0 or i == len(metrics.training_history) - 1:
                milestones.append(entry)

        # Limit to reasonable number for report (first 10, middle 5, last 10)
        if len(milestones) > 25:
            selected = milestones[:10] + milestones[len(milestones)//2-2:len(milestones)//2+3] + milestones[-10:]
        else:
            selected = milestones

        section = """## Training History

### Training Progress (Key Steps)

| Step | Epoch | Loss | Learning Rate | Grad Norm |
|------|-------|------|---------------|-----------|
"""

        for entry in selected:
            step = entry.get('step', 0)
            epoch = entry.get('epoch', 0)
            loss = entry.get('loss', 0)
            lr = entry.get('learning_rate', 0)
            grad_norm = entry.get('grad_norm', 0)

            section += f"| {step} | {epoch:.2f} | {loss:.4f} | {lr:.2e} | {grad_norm:.2f} |\n"

        # Add summary statistics
        all_losses = [e.get('loss', 0) for e in metrics.training_history if 'loss' in e]
        all_grads = [e.get('grad_norm', 0) for e in metrics.training_history if 'grad_norm' in e]

        if all_losses:
            mean_grad = np.mean(all_grads) if all_grads else 0
            std_grad = np.std(all_grads) if all_grads else 0

            section += f"""
### Training Summary
- **Initial Loss**: {all_losses[0]:.4f}
- **Final Loss**: {all_losses[-1]:.4f}
- **Loss Improvement**: {all_losses[0] - all_losses[-1]:+.4f}
- **Mean Gradient Norm**: {mean_grad:.2f} (σ={std_grad:.2f})
- **Total Training Steps**: {len(metrics.training_history)}
"""

        return section

    def _create_results_summary(self, analysis: Dict[str, Any]) -> str:
        """Create results summary section."""
        min_general = analysis["min_general_loss"]
        zero_crossings = analysis["zero_crossings"]

        zc_section = ""
        if zero_crossings:
            zc_section = "\n### Zero-Crossings Found ★\n\n"
            for i, result in enumerate(zero_crossings[:3], 1):
                zc_section += f"{i}. α = {result.alpha:+.4f}, |ΔL| = {result.functional_return:.6f}\n"

        return f"""## Results Summary

### Minimum General Loss
- **α**: {min_general.alpha:+.4f}
- **Loss**: {min_general.loss:.4f}
- **Δ from base**: {min_general.loss - min_general.base_loss:+.4f}
{zc_section}"""

    def _create_alpha_sweep_details(
        self,
        results: List[AlphaSweepResult],
        metrics: ExperimentMetrics
    ) -> str:
        """Create alpha sweep details section with sample data points.

        Args:
            results: List of alpha sweep results
            metrics: Experiment metrics

        Returns:
            Alpha sweep details section as string
        """
        if not results:
            return """## Alpha Sweep Details

**No alpha sweep results available.**"""

        # Select representative samples (evenly spaced)
        num_samples = min(20, len(results))
        step = max(1, len(results) // num_samples)  # Guard against division issues
        selected = [results[i * step] for i in range(num_samples) if i * step < len(results)]
        if results[-1] not in selected:
            selected.append(results[-1])

        section = f"""## Alpha Sweep Details

**Base Model Loss**: L(M_base) = {results[0].base_loss:.4f}

### Sample Data Points

| α | L(α) | L(2α) | |ΔL| | |ΔL(2α)| | Task L(α) | PPL(α) | PPL(2α) |
|---|------|-------|--------|------------|-----------|--------|---------|
"""

        for result in selected:
            # Calculate perplexities if not already set
            perplexity = result.perplexity if result.perplexity > 0 else np.exp(result.loss)
            perplexity_2alpha = result.perplexity_2alpha if result.perplexity_2alpha > 0 else np.exp(result.loss_2alpha)

            section += f"| {result.alpha:+.3f} | {result.loss:.4f} | {result.loss_2alpha:.4f} | "
            section += f"{result.functional_return:.6f} | {result.functional_return_2alpha:.6f} | "
            section += f"{result.task_performance:.4f} | {perplexity:.2f} | {perplexity_2alpha:.2f} |\n"

        # Calculate step size safely
        alpha_step = 0
        if metrics.num_alpha_samples > 1:
            alpha_step = (metrics.alpha_range[1] - metrics.alpha_range[0]) / (metrics.num_alpha_samples - 1)

        section += f"""
### Key Metrics
- **Total Samples**: {len(results)}
- **Alpha Range**: [{metrics.alpha_range[0]:.1f}, {metrics.alpha_range[1]:.1f}]
- **Alpha Step Size**: {alpha_step:.4f}
- **Total Sweep Time**: {metrics.sweep_duration_seconds / 60:.1f} minutes
- **Avg Time per Sample**: {metrics.time_per_alpha_seconds:.2f}s
"""

        return section

    def _create_statistical_summary(
        self,
        results: List[AlphaSweepResult],
        analysis: Dict[str, Any]
    ) -> str:
        """Create statistical summary section.

        Args:
            results: List of alpha sweep results
            analysis: Analysis dictionary

        Returns:
            Statistical summary section as string
        """
        # Guard against empty results
        if not results:
            return """## Statistical Summary

**No results available for statistical analysis.**"""

        # Calculate statistics
        all_losses = [r.loss for r in results]
        all_functional_returns = [r.functional_return for r in results]
        all_task_perfs = [r.task_performance for r in results]

        # Guard against empty arrays (shouldn't happen if results is non-empty, but be safe)
        if not all_losses or not all_functional_returns or not all_task_perfs:
            return """## Statistical Summary

**Incomplete data for statistical analysis.**"""

        min_loss_idx = np.argmin(all_losses)
        max_loss_idx = np.argmax(all_losses)
        min_fr_idx = np.argmin(all_functional_returns)
        max_fr_idx = np.argmax(all_functional_returns)
        min_task_idx = np.argmin(all_task_perfs)
        max_task_idx = np.argmax(all_task_perfs)

        return f"""## Statistical Summary

### Loss Distribution (General)
- **Mean**: {np.mean(all_losses):.4f}
- **Std Dev**: {np.std(all_losses):.4f}
- **Min**: {np.min(all_losses):.4f} (at α = {results[min_loss_idx].alpha:+.4f})
- **Max**: {np.max(all_losses):.4f} (at α = {results[max_loss_idx].alpha:+.4f})

### Functional Return Distribution
- **Mean**: {np.mean(all_functional_returns):.4f}
- **Std Dev**: {np.std(all_functional_returns):.4f}
- **Min**: {np.min(all_functional_returns):.6f} (at α = {results[min_fr_idx].alpha:+.4f})
- **Max**: {np.max(all_functional_returns):.4f} (at α = {results[max_fr_idx].alpha:+.4f})

### Task Performance Distribution
- **Mean**: {np.mean(all_task_perfs):.4f}
- **Std Dev**: {np.std(all_task_perfs):.4f}
- **Min**: {np.min(all_task_perfs):.4f} (at α = {results[min_task_idx].alpha:+.4f})
- **Max**: {np.max(all_task_perfs):.4f} (at α = {results[max_task_idx].alpha:+.4f})"""

    def _create_category_breakdown(
        self,
        results: List[AlphaSweepResult],
        metrics: ExperimentMetrics
    ) -> str:
        """Create per-category loss breakdown section.

        This section shows how the task vector affects different domains
        when using the "combined" evaluation dataset.

        Args:
            results: List of alpha sweep results with category losses
            metrics: Experiment metrics

        Returns:
            Category breakdown section as string
        """
        # Get all unique categories from the first result
        if not results or not results[0].category_losses:
            return ""

        categories = sorted(results[0].category_losses.keys())

        # Find best alpha for each category
        best_alphas = {}
        for category in categories:
            category_losses_at_alpha = [(r.alpha, r.category_losses.get(category, float('inf'))) for r in results]
            best_alpha, best_loss = min(category_losses_at_alpha, key=lambda x: x[1])
            best_alphas[category] = (best_alpha, best_loss)

        # Build the section
        section = f"""## Per-Category Loss Analysis

**Dataset**: {metrics.general_eval_dataset}

This breakdown shows how the task vector ({metrics.task_name}) affects each evaluation domain separately.

### Best α for Each Category

| Category | Best α | Loss at Best α | Interpretation |
|----------|--------|----------------|----------------|
"""

        interpretations = {
            "coding": "How well does the model handle programming/technical content?",
            "wikitext": "How well does the model handle factual/encyclopedic content?",
            "mixed_domain": "How well does the model handle diverse multi-domain content?",
            "common_knowledge": "How well does the model handle everyday general knowledge?"
        }

        for category in categories:
            best_alpha, best_loss = best_alphas[category]
            interp = interpretations.get(category, "Domain-specific performance")
            section += f"| {category} | {best_alpha:+.4f} | {best_loss:.4f} | {interp} |\n"

        # Add category-wise comparison at α=0 (base model)
        base_result = min(results, key=lambda r: abs(r.alpha))  # Find closest to α=0
        section += f"\n### Baseline Comparison (α ≈ {base_result.alpha:.3f})\n\n"
        section += "Loss by category before applying task vector:\n\n"
        section += "| Category | Loss |\n"
        section += "|----------|------|\n"

        for category in categories:
            loss = base_result.category_losses.get(category, 0.0)
            section += f"| {category} | {loss:.4f} |\n"

        section += "\n**Insight**: Lower values indicate better performance in that domain."

        return section

    def _create_squaring_test_analysis(self, analysis: Dict[str, Any]) -> str:
        """Create squaring test analysis section.

        Args:
            analysis: Analysis dictionary

        Returns:
            Squaring test analysis section as string
        """
        squaring_returns = analysis.get("squaring_return_points", [])

        section = """## Squaring Test Analysis: [W(λ)]² = I Analog

This experiment tests whether neural loss landscapes exhibit rotation-like symmetry properties
analogous to the [W(λ)]² = I property in rotation groups (Eckmann & Tlusty, 2025).

We evaluate both L(α) and L(2α) at each α value to identify "squaring return points" where
doubling the task vector scaling returns the loss to approximately the base model level.

"""

        if squaring_returns:
            section += f"""### Squaring Return Points Found ★

Found {len(squaring_returns)} α value(s) where L(2α) ≈ L(M_base):

| # | α | L(2α) | |L(2α) - L_base| |
|---|---|-------|----------------|
"""
            for i, sr in enumerate(squaring_returns[:10], 1):
                section += f"| {i} | {sr.alpha:+.4f} | {sr.loss_2alpha:.4f} | {sr.functional_return_2alpha:.6f} |\n"

            section += """
**Interpretation**: These α values suggest special scaling factors where doubling the task
vector exhibits self-inverse-like properties. This suggests neural loss landscapes MAY exhibit
rotation-like symmetry!

**Connection to paper**: Analogous to the 180° rotations in SO(3) that square to identity,
these α values represent "functional return" points under doubling in the neural loss landscape.
"""
        else:
            section += """### No Squaring Return Points Found

The loss landscape does not exhibit squaring return properties under doubling the task vector
scaling. L(2α) does not return to approximately L(M_base) for any α ≠ 0.

**Interpretation**: Unlike rotation groups where [W(λ)]² = I has abundant solutions, the
neural loss landscape does not show similar self-inverse properties under task vector doubling.
"""

        return section

    def _create_2d_composition_section(
        self,
        results_2d: List[TwoDSweepResult],
        metrics: ExperimentMetrics
    ) -> str:
        """Create 2D composition analysis section.

        Args:
            results_2d: List of 2D composition results
            metrics: Experiment metrics

        Returns:
            2D composition analysis section as string
        """
        # Extract alpha and beta ranges from results
        alphas = sorted(set(r.alpha for r in results_2d))
        betas = sorted(set(r.beta for r in results_2d))
        alpha_min, alpha_max = min(alphas), max(alphas)
        beta_min, beta_max = min(betas), max(betas)
        grid_size = f"{len(alphas)}×{len(betas)}"
        total_evaluations = len(results_2d)

        # Find minimum and maximum loss points
        min_result = min(results_2d, key=lambda r: r.loss)
        max_result = max(results_2d, key=lambda r: r.loss)

        # Find point closest to base loss (functional return)
        base_loss = results_2d[0].base_loss
        closest_to_base = min(results_2d, key=lambda r: r.functional_return)

        # Find points along axes (α=0 and β=0)
        alpha_axis_results = [r for r in results_2d if abs(r.beta) < 0.01]
        beta_axis_results = [r for r in results_2d if abs(r.alpha) < 0.01]

        section = f"""## 2D Task Vector Composition Analysis

This experiment explores the loss landscape under composition of two task vectors:
**L(M_base + α·T1 + β·T2)**

### Experiment Setup

- **First Task Vector (T1)**: {metrics.task_name}
  - Magnitude: ||T1|| = {metrics.task_vector_magnitude:.4f}
- **Second Task Vector (T2)**: sentiment_negative
  - Magnitude: ||T2|| = {metrics.task_vector_2_magnitude:.4f}
- **Grid Configuration**: {grid_size} = {total_evaluations} evaluations
- **α range**: [{alpha_min:.1f}, {alpha_max:.1f}]
- **β range**: [{beta_min:.1f}, {beta_max:.1f}]
- **Base Model Loss**: L(M_base) = {base_loss:.4f}

### Key Findings

#### Minimum Loss (Optimal Composition)
- **Location**: (α = {min_result.alpha:+.4f}, β = {min_result.beta:+.4f})
- **Loss**: {min_result.loss:.4f}
- **Improvement over base**: {base_loss - min_result.loss:+.4f}
- **Perplexity**: {min_result.perplexity:.2f}

#### Maximum Loss (Worst Composition)
- **Location**: (α = {max_result.alpha:+.4f}, β = {max_result.beta:+.4f})
- **Loss**: {max_result.loss:.4f}
- **Degradation from base**: {max_result.loss - base_loss:+.4f}
- **Perplexity**: {max_result.perplexity:.2f}

#### Closest Return to Base (Functional Return)
- **Location**: (α = {closest_to_base.alpha:+.4f}, β = {closest_to_base.beta:+.4f})
- **Loss**: {closest_to_base.loss:.4f}
- **|L - L_base|**: {closest_to_base.functional_return:.6f}
"""

        # Add axis analysis if we have axis results
        if alpha_axis_results:
            min_alpha_axis = min(alpha_axis_results, key=lambda r: r.loss)
            section += f"""
#### Along α-axis (β ≈ 0, pure T1 scaling)
- **Best α**: {min_alpha_axis.alpha:+.4f}
- **Loss**: {min_alpha_axis.loss:.4f}
"""

        if beta_axis_results:
            min_beta_axis = min(beta_axis_results, key=lambda r: r.loss)
            section += f"""
#### Along β-axis (α ≈ 0, pure T2 scaling)
- **Best β**: {min_beta_axis.beta:+.4f}
- **Loss**: {min_beta_axis.loss:.4f}
"""

        # Add loss distribution statistics
        all_losses = [r.loss for r in results_2d]
        all_functional_returns = [r.functional_return for r in results_2d]

        section += f"""
### Loss Landscape Statistics

- **Mean Loss**: {np.mean(all_losses):.4f}
- **Std Dev**: {np.std(all_losses):.4f}
- **Loss Range**: [{np.min(all_losses):.4f}, {np.max(all_losses):.4f}]
- **Mean Functional Return**: {np.mean(all_functional_returns):.4f}

### Interpretation

The 2D composition experiment reveals how two task vectors interact when combined.
The loss landscape shows whether:

1. **Additive effects**: Task vectors combine linearly (smooth gradients)
2. **Synergistic effects**: Certain combinations perform better than either task alone
3. **Interference effects**: Task vectors cancel or degrade performance when combined
4. **Rotation-like patterns**: Circular or symmetric patterns suggesting geometric structure

**Visualization**: See `loss_landscape_2d.png` for the complete heatmap showing L(α,β) across
the entire grid. The heatmap uses color intensity to show loss values, with the base model
marked at the origin (α=0, β=0).

**Data**: Full numerical results available in `loss_landscape_2d_results.json`
"""

        return section

    def _create_theoretical_connection(self, analysis: Dict[str, Any]) -> str:
        """Create theoretical connection section.

        Args:
            analysis: Analysis dictionary

        Returns:
            Theoretical connection section as string
        """
        zero_crossings = analysis["zero_crossings"]
        has_zc = len(zero_crossings) > 0

        # Check for squaring return points
        squaring_returns = analysis.get("squaring_return_points", [])
        has_squaring = len(squaring_returns) > 0

        # Determine finding based on both zero-crossings and squaring returns
        if has_zc and has_squaring:
            finding = "Yes - found both zero-crossings and squaring return points suggesting rotation-like symmetry"
        elif has_squaring:
            finding = "Partial - found squaring return points where L(2α) ≈ L_base, suggesting potential rotation-like symmetry"
        elif has_zc:
            finding = "Partial - found zero-crossings suggesting special α values"
        else:
            finding = "No - loss is monotonic, no rotation-like symmetry detected"

        return f"""## Connection to Theoretical Background

**Paper Reference**: Eckmann & Tlusty (2025), "Walks in Rotation Spaces Return Home when Doubled and Scaled"
(arXiv:2502.14367v3)

### Key Theorem

The paper proves that for rotation groups SO(3)/SU(2), almost any walk W can be scaled to reach
a 180° rotation R(n,π), which when squared returns to identity: R(n,π)² = I. This property is
abundant (density 2/π ≈ 64%).

### Our Experiment

**Rotation groups vs Task vectors:**
- **Rotation groups**: Multiplicative group with composition W₁ ∘ W₂
- **Task vectors**: Additive vector space with addition v₁ + v₂
- **Key difference**: Task vectors lack the group structure required for the theorem

**Question**: Do neural loss landscapes exhibit analogous "functional return" properties under scaling?

**Finding**: {finding}

### Implications

The task vector approach explores whether the geometric properties proven for rotation groups
have analogs in the parameter space of neural networks. While task vectors lack the formal group
structure, identifying special scaling factors (zero-crossings, squaring returns) suggests that
loss landscapes may exhibit similar functional symmetries.

This connection opens interesting questions about the geometry of neural network parameter spaces
and whether principles from group theory and differential geometry can inform model merging and
task composition strategies."""

    def _create_recommendations(self, analysis: Dict[str, Any]) -> str:
        """Create practical recommendations section."""
        min_general = analysis["min_general_loss"]
        min_task = analysis["min_task_loss"]

        return f"""## Practical Recommendations

1. **For General Knowledge**: Use α = {min_general.alpha:+.4f}
2. **For Task Performance**: Use α = {min_task.alpha:+.4f}
3. **For Model Merging**: Consider alpha values near zero-crossings

---
*Generated by SITV - Self-Inverse Task Vectors*"""
