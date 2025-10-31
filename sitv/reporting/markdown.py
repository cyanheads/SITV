"""
Markdown report generation for SITV experiments.

This module provides the MarkdownReportGenerator for creating comprehensive
experiment reports in Markdown format for LLM analysis.
"""

import json
from typing import Dict, Any, List
from datetime import datetime
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

    # TODO: Migrate remaining sections from main.py:generate_markdown_report()
    # - Statistical analysis section
    # - Loss landscape interpretation
    # - Squaring test analysis
    # - Theoretical connection section
    # - Full formatting with proper markdown tables
