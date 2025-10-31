"""
Markdown report generation for SITV experiments.

This module provides the MarkdownReportGenerator for creating comprehensive
experiment reports in Markdown format for LLM analysis.
"""

import json
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from sitv.data.models import AlphaSweepResult, ExperimentMetrics


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
        output_path: str = "experiment_report.md"
    ) -> str:
        """Generate comprehensive Markdown report.

        Args:
            results: List of alpha sweep results
            analysis: Analysis dictionary from ResultAnalyzer
            metrics: ExperimentMetrics object with full experiment data
            output_path: Path to save report

        Returns:
            Path to saved report

        Examples:
            >>> generator = MarkdownReportGenerator()
            >>> report_path = generator.generate(results, analysis, metrics)
        """
        report = self._build_report(results, analysis, metrics)

        # Write report
        with open(output_path, 'w') as f:
            f.write(report)

        print(f"\nMarkdown report saved: {output_path}")
        return output_path

    def _build_report(
        self,
        results: List[AlphaSweepResult],
        analysis: Dict[str, Any],
        metrics: ExperimentMetrics
    ) -> str:
        """Build the complete report content.

        Args:
            results: List of results
            analysis: Analysis dictionary
            metrics: Experiment metrics

        Returns:
            Complete Markdown report as string
        """
        sections = []

        sections.append(self._create_header(metrics))
        sections.append(self._create_executive_summary(analysis, metrics))
        sections.append(self._create_configuration_section(metrics))
        sections.append(self._create_timing_breakdown(metrics))
        sections.append(self._create_results_summary(analysis))
        sections.append(self._create_statistical_summary(results, analysis))

        # Add squaring test analysis if available
        if analysis.get("has_squaring_data", False):
            sections.append(self._create_squaring_test_analysis(analysis))

        sections.append(self._create_theoretical_connection(analysis))
        sections.append(self._create_recommendations(analysis))

        return "\n\n".join(sections)

    def _create_header(self, metrics: ExperimentMetrics) -> str:
        """Create report header."""
        return f"""# SITV Experiment Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Model**: {metrics.model_name}
**Task**: {metrics.task_name}
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

        return f"""## Executive Summary

- **Best General α**: {min_general.alpha:+.4f} (Loss: {min_general.loss:.4f})
- **Best Task α**: {min_task.alpha:+.4f} (Loss: {min_task.task_performance:.4f})
- **Zero-Crossings**: {zero_crossings} found
- **Total Duration**: {metrics.duration_seconds / 60:.1f} minutes"""

    def _create_configuration_section(self, metrics: ExperimentMetrics) -> str:
        """Create configuration section."""
        return f"""## Configuration

- **Alpha Range**: [{metrics.alpha_range[0]}, {metrics.alpha_range[1]}]
- **Samples**: {metrics.num_alpha_samples}
- **Training Examples**: {metrics.training_examples}
- **Epochs**: {metrics.num_epochs}
- **Learning Rate**: {metrics.learning_rate}"""

    def _create_timing_breakdown(self, metrics: ExperimentMetrics) -> str:
        """Create timing breakdown section."""
        ft_pct = (metrics.finetuning_duration_seconds / metrics.duration_seconds * 100) if metrics.duration_seconds > 0 else 0
        sweep_pct = (metrics.sweep_duration_seconds / metrics.duration_seconds * 100) if metrics.duration_seconds > 0 else 0

        return f"""## Timing Breakdown

| Phase | Duration | Percentage |
|-------|----------|------------|
| Fine-tuning | {metrics.finetuning_duration_seconds / 60:.1f}m | {ft_pct:.1f}% |
| Alpha Sweep | {metrics.sweep_duration_seconds / 60:.1f}m | {sweep_pct:.1f}% |
| **Total** | **{metrics.duration_seconds / 60:.1f}m** | **100%** |"""

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
        # Calculate statistics
        all_losses = [r.loss for r in results]
        all_functional_returns = [r.functional_return for r in results]
        all_task_perfs = [r.task_performance for r in results]

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

    def _create_theoretical_connection(self, analysis: Dict[str, Any]) -> str:
        """Create theoretical connection section.

        Args:
            analysis: Analysis dictionary

        Returns:
            Theoretical connection section as string
        """
        zero_crossings = analysis["zero_crossings"]
        has_zc = len(zero_crossings) > 0

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

**Finding**: {"Yes - found zero-crossings suggesting special α values" if has_zc else "No - loss is monotonic, no rotation-like symmetry detected"}

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
