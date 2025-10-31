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
        """Plot 1D alpha sweep results with optional squaring test.

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
        functional_returns = [r.functional_return for r in results]
        task_perfs = [r.task_eval_loss for r in results]

        # Extract squaring test data if available
        if enable_squaring_test and analysis.get("has_squaring_data", False):
            losses_2alpha = [r.loss_2alpha for r in results]
            functional_returns_2alpha = [r.functional_return_2alpha for r in results]

        # Create figure with appropriate grid layout
        if enable_squaring_test:
            fig, axes = plt.subplots(2, 3, figsize=(21, 10))
            title = "Task Vector Loss Landscape: L(M_base + αT) with Squaring Test\n" \
                    "Inspired by Eckmann & Tlusty (2025)"
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            title = "Task Vector Loss Landscape: L(M_base + αT)\n" \
                    "Inspired by Eckmann & Tlusty (2025)"

        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

        # ─── Plot 1: MAIN - Loss vs α (KEY PLOT!) ───
        axes[0, 0].plot(alphas, losses, 'b-', linewidth=2.5, label='General Loss', alpha=0.8)
        axes[0, 0].plot(alphas, task_perfs, 'g-', linewidth=2.0, label='Task Loss', alpha=0.6)
        axes[0, 0].axhline(y=base_loss, color='red', linestyle='--', linewidth=2,
                           label='L(M_base)', alpha=0.7)
        axes[0, 0].axvline(x=0, color='gray', linestyle='--', alpha=0.4)

        # Highlight zero-crossings
        if analysis['zero_crossings']:
            zc_alphas = [r.alpha for r in analysis['zero_crossings']]
            zc_losses = [r.loss for r in analysis['zero_crossings']]
            axes[0, 0].scatter(zc_alphas, zc_losses, color='orange', s=150, zorder=5,
                              marker='*', edgecolors='black', linewidth=1,
                              label='Zero-crossings')

        # Highlight general minimum
        min_gen_result = analysis['min_general_loss']
        axes[0, 0].scatter([min_gen_result.alpha], [min_gen_result.loss],
                          color='blue', s=150, zorder=5, marker='D',
                          edgecolors='black', linewidth=1,
                          label=f'Min General (α={min_gen_result.alpha:.2f})')

        # Highlight task minimum
        min_task_result = analysis['min_task_loss']
        axes[0, 0].scatter([min_task_result.alpha], [min_task_result.task_eval_loss],
                          color='green', s=150, zorder=5, marker='D',
                          edgecolors='black', linewidth=1,
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
            axes[0, 1].scatter(zc_alphas, zc_returns, color='green', s=150, zorder=5,
                              marker='*', edgecolors='black', linewidth=1)

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
            axes[1, 0].scatter(zc_alphas, zc_deltas, color='green', s=150, zorder=5,
                              marker='*', edgecolors='black', linewidth=1)

        axes[1, 0].set_xlabel('α', fontsize=12)
        axes[1, 0].set_ylabel('L(α) - L(M_base)', fontsize=12)
        axes[1, 0].set_title('Signed Loss Delta', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # ─── Plot 4: Task Performance vs α ───
        axes[1, 1].plot(alphas, task_perfs, 'g-', linewidth=2, alpha=0.8, label='Task Loss')
        axes[1, 1].axhline(y=base_loss, color='gray', linestyle='--', alpha=0.5,
                          label='Base Loss')
        axes[1, 1].axvline(x=0, color='gray', linestyle='--', alpha=0.4)

        axes[1, 1].set_xlabel('α', fontsize=12)
        axes[1, 1].set_ylabel('Task-Specific Loss', fontsize=12)
        axes[1, 1].set_title('Task Performance', fontsize=12, fontweight='bold')
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)

        # ─── Squaring Test Plots (Enhancement #3) ───
        if enable_squaring_test and analysis.get("has_squaring_data", False):
            # Plot 5: Squaring Test - L(α) vs L(2α)
            axes[0, 2].plot(alphas, losses, 'b-', linewidth=2.5, label='L(α)', alpha=0.8)
            axes[0, 2].plot(alphas, losses_2alpha, 'r-', linewidth=2.5, label='L(2α)', alpha=0.8)
            axes[0, 2].axhline(y=base_loss, color='green', linestyle='--', linewidth=2,
                              label='L(M_base)', alpha=0.7)
            axes[0, 2].axvline(x=0, color='gray', linestyle='--', alpha=0.4)

            # Highlight squaring return points (where L(2α) ≈ L_base)
            if 'squaring_return_points' in analysis:
                sr_alphas = [r.alpha for r in analysis['squaring_return_points']]
                sr_losses_2alpha = [r.loss_2alpha for r in analysis['squaring_return_points']]
                axes[0, 2].scatter(sr_alphas, sr_losses_2alpha, color='orange', s=150,
                                  zorder=5, marker='*', edgecolors='black', linewidth=1,
                                  label='Squaring returns')

            axes[0, 2].set_xlabel('α', fontsize=12)
            axes[0, 2].set_ylabel('Loss', fontsize=12)
            axes[0, 2].set_title('Squaring Test: [W(λ)]² = I Analog', fontsize=12, fontweight='bold')
            axes[0, 2].legend(fontsize=8, loc='best')
            axes[0, 2].grid(True, alpha=0.3)

            # Plot 6: Squaring Functional Return |L(2α) - L_base|
            axes[1, 2].plot(alphas, functional_returns_2alpha, 'r-', linewidth=2.5,
                           alpha=0.8, label='|L(2α) - L_base|')
            axes[1, 2].plot(alphas, functional_returns, 'b--', linewidth=1.5, alpha=0.5,
                           label='|L(α) - L_base|')
            axes[1, 2].axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
            axes[1, 2].axvline(x=0, color='gray', linestyle='--', alpha=0.4)

            # Highlight squaring return points
            if 'squaring_return_points' in analysis:
                sr_alphas = [r.alpha for r in analysis['squaring_return_points']]
                sr_returns_2alpha = [r.functional_return_2alpha for r in analysis['squaring_return_points']]
                axes[1, 2].scatter(sr_alphas, sr_returns_2alpha, color='orange', s=150,
                                  zorder=5, marker='*', edgecolors='black', linewidth=1)

            axes[1, 2].set_xlabel('α', fontsize=12)
            axes[1, 2].set_ylabel('|L(2α) - L(M_base)|', fontsize=12)
            axes[1, 2].set_title('Squaring Functional Return', fontsize=12, fontweight='bold')
            axes[1, 2].legend(fontsize=8, loc='best')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n📊 Plot saved to {output_path}")
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
            extent=(min(alphas), max(alphas), min(betas), max(betas)),
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
