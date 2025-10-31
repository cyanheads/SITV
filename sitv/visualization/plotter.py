"""
Visualization service for SITV experiments.

This module provides the ResultPlotter for creating plots of experiment results,
including 1D alpha sweeps and 2D composition heatmaps.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from sitv.data.models import AlphaSweepResult, TwoDSweepResult


class ResultPlotter:
    """Service for plotting experiment results.

    This plotter creates visualizations for:
    - 1D alpha sweep: Loss vs alpha curves
    - 2D composition: Heatmaps of loss landscapes
    - Squaring test comparisons
    - Zero-crossing markers

    NOTE: This is a refactored version. The original 238-line plot_results()
    function from main.py should be fully migrated here. For now, this provides
    the essential structure with simplified plotting.

    Attributes:
        figsize: Default figure size
        dpi: Default DPI for saved figures
    """

    def __init__(self, figsize: tuple = (16, 10), dpi: int = 100):
        """Initialize the result plotter.

        Args:
            figsize: Figure size (width, height)
            dpi: DPI for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi

    def plot_alpha_sweep(
        self,
        results: List[AlphaSweepResult],
        analysis: Dict[str, Any],
        output_path: str = "loss_landscape_sweep.png",
        enable_squaring_test: bool = False
    ) -> str:
        """Plot 1D alpha sweep results.

        Args:
            results: List of AlphaSweepResult objects
            analysis: Analysis dictionary from ResultAnalyzer
            output_path: Path to save plot
            enable_squaring_test: Whether to include squaring test plots

        Returns:
            Path to saved plot

        Examples:
            >>> plotter = ResultPlotter()
            >>> plot_path = plotter.plot_alpha_sweep(results, analysis)
        """
        # Extract data
        alphas = [r.alpha for r in results]
        losses = [r.loss for r in results]
        base_loss = results[0].base_loss if results else 0

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Loss Landscape: L(M_base + α·T)', fontsize=16, fontweight='bold')

        # Plot 1: Loss vs Alpha
        ax = axes[0, 0]
        ax.plot(alphas, losses, 'b-', linewidth=2, label='L(α)')
        ax.axhline(y=base_loss, color='r', linestyle='--', label='L(M_base)')
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('α')
        ax.set_ylabel('Loss')
        ax.set_title('General Loss Landscape')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Mark minimum
        min_result = analysis['min_general_loss']
        ax.plot(min_result.alpha, min_result.loss, 'g*', markersize=15, label='Min')

        # Mark zero-crossings
        for zc in analysis['zero_crossings']:
            ax.plot(zc.alpha, zc.loss, 'r*', markersize=12)

        # Plot 2: Functional Return
        ax = axes[0, 1]
        functional_returns = [r.functional_return for r in results]
        ax.plot(alphas, functional_returns, 'purple', linewidth=2)
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('α')
        ax.set_ylabel('|L(α) - L(M_base)|')
        ax.set_title('Functional Return')
        ax.grid(True, alpha=0.3)

        # Plot 3: Task Performance
        ax = axes[1, 0]
        task_perf = [r.task_performance for r in results]
        ax.plot(alphas, task_perf, 'orange', linewidth=2, label='Task Loss')
        ax.axhline(y=base_loss, color='r', linestyle='--', label='Base Loss')
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('α')
        ax.set_ylabel('Loss')
        ax.set_title('Task-Specific Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Perplexity
        ax = axes[1, 1]
        perplexities = [r.perplexity for r in results]
        ax.plot(alphas, perplexities, 'green', linewidth=2)
        ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('α')
        ax.set_ylabel('Perplexity')
        ax.set_title('Perplexity vs Alpha')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Plot saved: {output_path}")
        return output_path

    def plot_2d_composition(
        self,
        results: List[TwoDSweepResult],
        output_path: str = "loss_landscape_2d.png"
    ) -> str:
        """Plot 2D composition heatmap.

        Args:
            results: List of TwoDSweepResult objects
            output_path: Path to save plot

        Returns:
            Path to saved plot

        Examples:
            >>> plotter = ResultPlotter()
            >>> plot_path = plotter.plot_2d_composition(results_2d)
        """
        # Extract unique alpha and beta values
        alphas = sorted(set(r.alpha for r in results))
        betas = sorted(set(r.beta for r in results))

        # Create loss matrix
        loss_matrix = np.zeros((len(betas), len(alphas)))
        for r in results:
            i = betas.index(r.beta)
            j = alphas.index(r.alpha)
            loss_matrix[i, j] = r.loss

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot heatmap
        im = ax.imshow(
            loss_matrix,
            extent=[min(alphas), max(alphas), min(betas), max(betas)],
            origin='lower',
            aspect='auto',
            cmap='viridis'
        )

        ax.set_xlabel('α (Task Vector 1)')
        ax.set_ylabel('β (Task Vector 2)')
        ax.set_title('2D Loss Landscape: L(M_base + α·T1 + β·T2)')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Loss')

        # Mark origin
        ax.plot(0, 0, 'r*', markersize=20, label='Base Model')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"2D plot saved: {output_path}")
        return output_path

    # TODO: Migrate remaining visualization features from main.py:
    # - Multi-panel layouts with more detailed plots
    # - Squaring test comparison plots
    # - Statistical distribution plots
    # - Enhanced annotations and markers
    # - Custom color schemes and styling
