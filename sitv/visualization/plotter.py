"""
Visualization service for SITV experiments.

This module provides the ResultPlotter for creating plots of experiment results,
including 1D alpha sweeps and 2D composition heatmaps.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
from sitv.data.models import AlphaSweepResult, TwoDSweepResult, ThreeDSweepResult


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

    def plot_3d_composition(
        self,
        results: List[ThreeDSweepResult],
        output_path_interactive: str = "loss_landscape_3d_interactive.html",
        output_path_slices: str = "loss_landscape_3d_slices.png",
        threshold: Optional[float] = None
    ) -> tuple[str, str]:
        """Plot 3D composition results with interactive volume and 2D slices.

        Creates two visualizations:
        1. Interactive 3D scatter/isosurface plot (HTML via plotly)
        2. 2D slice views through the 3D volume (PNG via matplotlib)

        Args:
            results: List of ThreeDSweepResult objects
            output_path_interactive: Path to save interactive HTML plot
            output_path_slices: Path to save 2D slices plot
            threshold: Loss threshold for isosurface (default: None = auto)

        Returns:
            Tuple of (interactive_plot_path, slices_plot_path)

        Examples:
            >>> plotter = ResultPlotter()
            >>> html_path, png_path = plotter.plot_3d_composition(results_3d)
        """
        try:
            import plotly.graph_objects as go  # type: ignore[import-untyped]
            plotly_available = True
        except ImportError:
            plotly_available = False
            print("⚠️  Warning: plotly not available, skipping interactive 3D plot")
            print("   Install with: pip install plotly")

        # Extract data
        alphas = np.array([r.alpha for r in results])
        betas = np.array([r.beta for r in results])
        gammas = np.array([r.gamma for r in results])
        losses = np.array([r.loss for r in results])
        base_loss = results[0].base_loss if results else 0

        # Find optimal point
        min_idx = np.argmin(losses)
        optimal_alpha = alphas[min_idx]
        optimal_beta = betas[min_idx]
        optimal_gamma = gammas[min_idx]
        optimal_loss = losses[min_idx]

        print(f"\n3D Composition Optimal: (α={optimal_alpha:.4f}, β={optimal_beta:.4f}, γ={optimal_gamma:.4f})")
        print(f"  Loss = {optimal_loss:.4f}")

        # Create interactive 3D plot
        interactive_path = output_path_interactive
        if plotly_available:
            fig = go.Figure()

            # Add 3D scatter plot
            fig.add_trace(go.Scatter3d(
                x=alphas,
                y=betas,
                z=gammas,
                mode='markers',
                marker=dict(
                    size=3,
                    color=losses,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Loss"),
                    opacity=0.6
                ),
                text=[f'α={a:.2f}, β={b:.2f}, γ={g:.2f}<br>Loss={l:.4f}'
                      for a, b, g, l in zip(alphas, betas, gammas, losses)],
                hovertemplate='%{text}<extra></extra>',
                name='Loss Landscape'
            ))

            # Mark optimal point
            fig.add_trace(go.Scatter3d(
                x=[optimal_alpha],
                y=[optimal_beta],
                z=[optimal_gamma],
                mode='markers',
                marker=dict(size=10, color='red', symbol='diamond'),
                text=[f'Optimal<br>Loss={optimal_loss:.4f}'],
                hovertemplate='%{text}<extra></extra>',
                name='Optimal'
            ))

            # Mark origin (base model)
            fig.add_trace(go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode='markers',
                marker=dict(size=10, color='white', symbol='circle', line=dict(color='black', width=2)),
                text=[f'Base Model<br>Loss={base_loss:.4f}'],
                hovertemplate='%{text}<extra></extra>',
                name='Base Model'
            ))

            fig.update_layout(
                title='3D Loss Landscape: L(M_base + α·T1 + β·T2 + γ·T3)',
                scene=dict(
                    xaxis_title='α (Task Vector 1)',
                    yaxis_title='β (Task Vector 2)',
                    zaxis_title='γ (Task Vector 3)',
                ),
                width=1000,
                height=800,
            )

            fig.write_html(interactive_path)
            print(f"📊 Interactive 3D plot saved to {interactive_path}")
        else:
            interactive_path = ""

        # Create 2D slices through the 3D volume
        slices_path = self._plot_3d_slices(results, output_path_slices)

        return interactive_path, slices_path

    def _plot_3d_slices(
        self,
        results: List[ThreeDSweepResult],
        output_path: str
    ) -> str:
        """Plot 2D slices through the 3D loss landscape.

        Creates a grid of 2D heatmaps showing cross-sections through
        the 3D volume at different fixed values of γ.

        Args:
            results: List of ThreeDSweepResult objects
            output_path: Path to save the plot

        Returns:
            Path to saved plot
        """
        # Extract unique values
        alphas = sorted(set(r.alpha for r in results))
        betas = sorted(set(r.beta for r in results))
        gammas = sorted(set(r.gamma for r in results))

        # Select 6 representative gamma slices (or fewer if less data)
        num_slices = min(6, len(gammas))
        gamma_indices = np.linspace(0, len(gammas) - 1, num_slices, dtype=int)
        selected_gammas = [gammas[i] for i in gamma_indices]

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        base_loss = results[0].base_loss if results else 0

        for idx, gamma_value in enumerate(selected_gammas):
            # Create loss matrix for this gamma slice
            loss_matrix = np.full((len(betas), len(alphas)), np.nan)

            for r in results:
                if abs(r.gamma - gamma_value) < 0.01:  # tolerance for floating point
                    i = betas.index(min(betas, key=lambda x: abs(x - r.beta)))
                    j = alphas.index(min(alphas, key=lambda x: abs(x - r.alpha)))
                    loss_matrix[i, j] = r.loss

            # Plot heatmap
            im = axes[idx].imshow(
                loss_matrix,
                extent=(min(alphas), max(alphas), min(betas), max(betas)),
                origin='lower',
                aspect='auto',
                cmap='viridis',
                interpolation='bilinear'
            )

            axes[idx].set_xlabel('α', fontsize=10)
            axes[idx].set_ylabel('β', fontsize=10)
            axes[idx].set_title(f'γ = {gamma_value:.2f}', fontsize=11, fontweight='bold')

            # Mark origin
            axes[idx].plot(0, 0, 'r*', markersize=15, label='Base')

            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('Loss', fontsize=9)

        plt.suptitle('3D Loss Landscape: Cross-sections at different γ values',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"📊 3D slices plot saved to {output_path}")
        return output_path

    def plot_multi_task_comparison(
        self,
        task_datasets: Dict[str, Dict[str, Any]],
        output_path: str = "combined_task_comparison.png",
        layout: str = "overlaid"
    ) -> str:
        """Plot combined comparison of multiple task datasets.

        Args:
            task_datasets: Dictionary mapping task names to their data
                          Each value should contain 'results' and 'analysis' keys
            output_path: Path to save plot
            layout: Visualization layout ('overlaid', 'side_by_side', 'grid', 'publication', 'heatmap')

        Returns:
            Path to saved plot

        Examples:
            >>> plotter = ResultPlotter()
            >>> datasets = {
            ...     'SP': {'results': sp_results, 'analysis': sp_analysis},
            ...     'SN': {'results': sn_results, 'analysis': sn_analysis},
            ...     'QA': {'results': qa_results, 'analysis': qa_analysis}
            ... }
            >>> plotter.plot_multi_task_comparison(datasets, layout='overlaid')
        """
        # Color scheme for tasks
        colors = {
            'SP': '#2E86AB',  # Blue
            'SN': '#A23B72',  # Red/Magenta
            'QA': '#06A77D',  # Green
        }

        if layout == "overlaid":
            return self._plot_overlaid_comparison(task_datasets, output_path, colors)
        elif layout == "side_by_side":
            return self._plot_side_by_side_comparison(task_datasets, output_path, colors)
        elif layout == "grid":
            return self._plot_grid_comparison(task_datasets, output_path, colors)
        elif layout == "publication":
            return self._plot_publication_quality(task_datasets, output_path, colors)
        elif layout == "heatmap":
            return self._plot_heatmap_comparison(task_datasets, output_path, colors)
        else:
            raise ValueError(f"Unknown layout: {layout}")

    def _plot_overlaid_comparison(
        self,
        task_datasets: Dict[str, Dict[str, Any]],
        output_path: str,
        colors: Dict[str, str]
    ) -> str:
        """Create overlaid comparison (2×2 grid with all tasks on same plots)."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        fig.suptitle('Multi-Task Comparison: Loss Landscape Across Tasks\n'
                     'Inspired by Eckmann & Tlusty (2025)',
                     fontsize=14, fontweight='bold', y=0.995)

        # Panel 1: Overlaid Loss Curves
        ax = axes[0, 0]
        for task_name, data in task_datasets.items():
            results = data['results']
            alphas = [r.alpha for r in results]
            losses = [r.loss for r in results]
            base_loss = results[0].base_loss if results else 0

            ax.plot(alphas, losses, '-', linewidth=2.5,
                   color=colors.get(task_name, 'gray'),
                   label=f'{task_name}', alpha=0.8)

            # Mark optimal point
            analysis = data['analysis']
            min_result = analysis['min_general_loss']
            ax.scatter([min_result.alpha], [min_result.loss],
                      color=colors.get(task_name, 'gray'),
                      s=100, marker='D', edgecolors='black', linewidth=1, zorder=5)

        # Add base loss reference (using first dataset's base loss)
        first_data = list(task_datasets.values())[0]
        base_loss = first_data['results'][0].base_loss
        ax.axhline(y=base_loss, color='red', linestyle='--',
                  linewidth=2, label='L(M_base)', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

        ax.set_xlabel('α (Task Vector Scaling)', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Loss Landscape Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        # Panel 2: Functional Returns
        ax = axes[0, 1]
        for task_name, data in task_datasets.items():
            results = data['results']
            alphas = [r.alpha for r in results]
            returns = [r.functional_return for r in results]

            ax.plot(alphas, returns, '-', linewidth=2.5,
                   color=colors.get(task_name, 'gray'),
                   label=f'{task_name}', alpha=0.8)

            # Mark zero-crossings
            analysis = data['analysis']
            if analysis['zero_crossings']:
                zc_alphas = [r.alpha for r in analysis['zero_crossings']]
                zc_returns = [r.functional_return for r in analysis['zero_crossings']]
                ax.scatter(zc_alphas, zc_returns,
                          color=colors.get(task_name, 'gray'),
                          s=100, marker='*', edgecolors='black', linewidth=1, zorder=5)

        ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('α (Task Vector Scaling)', fontsize=11)
        ax.set_ylabel('|L(α) - L(M_base)|', fontsize=11)
        ax.set_title('Functional Return Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        # Panel 3: Task-Specific Performance
        ax = axes[1, 0]
        for task_name, data in task_datasets.items():
            results = data['results']
            alphas = [r.alpha for r in results]
            task_losses = [r.task_eval_loss for r in results]

            ax.plot(alphas, task_losses, '-', linewidth=2.5,
                   color=colors.get(task_name, 'gray'),
                   label=f'{task_name}', alpha=0.8)

            # Mark optimal task point
            analysis = data['analysis']
            min_task = analysis['min_task_loss']
            ax.scatter([min_task.alpha], [min_task.task_eval_loss],
                      color=colors.get(task_name, 'gray'),
                      s=100, marker='D', edgecolors='black', linewidth=1, zorder=5)

        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('α (Task Vector Scaling)', fontsize=11)
        ax.set_ylabel('Task-Specific Loss', fontsize=11)
        ax.set_title('Task Performance Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        # Panel 4: Comparative Statistics Table
        ax = axes[1, 1]
        ax.axis('off')

        # Build statistics table
        table_data = []
        headers = ['Task', 'Opt α', 'Min Loss', 'Δ from Base', 'Zero Cross.']

        for task_name, data in task_datasets.items():
            analysis = data['analysis']
            min_gen = analysis['min_general_loss']
            base_loss = analysis['min_general_loss'].base_loss
            delta = min_gen.loss - base_loss
            num_zc = len(analysis['zero_crossings'])

            table_data.append([
                task_name,
                f"{min_gen.alpha:.3f}",
                f"{min_gen.loss:.3f}",
                f"{delta:.3f}",
                str(num_zc)
            ])

        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.15, 0.15, 0.2, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Color code task names
        for i, task_name in enumerate(task_datasets.keys()):
            cell = table[(i+1, 0)]  # +1 to skip header row
            cell.set_facecolor(colors.get(task_name, 'white'))
            cell.set_text_props(color='white', weight='bold')

        # Style header
        for j in range(len(headers)):
            cell = table[(0, j)]
            cell.set_facecolor('#404040')
            cell.set_text_props(color='white', weight='bold')

        ax.set_title('Comparative Metrics', fontsize=12, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n📊 Combined plot saved to {output_path}")
        return output_path

    def _plot_side_by_side_comparison(
        self,
        task_datasets: Dict[str, Dict[str, Any]],
        output_path: str,
        colors: Dict[str, str]
    ) -> str:
        """Create side-by-side comparison (3×2 grid, one row per task)."""
        num_tasks = len(task_datasets)
        fig, axes = plt.subplots(num_tasks, 2, figsize=(14, 4.5 * num_tasks))

        if num_tasks == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('Side-by-Side Task Comparison: Loss Landscapes\n'
                     'Inspired by Eckmann & Tlusty (2025)',
                     fontsize=14, fontweight='bold', y=0.998)

        for idx, (task_name, data) in enumerate(task_datasets.items()):
            results = data['results']
            analysis = data['analysis']

            alphas = [r.alpha for r in results]
            losses = [r.loss for r in results]
            returns = [r.functional_return for r in results]
            base_loss = results[0].base_loss if results else 0

            color = colors.get(task_name, 'gray')

            # Left: Loss Landscape
            ax = axes[idx, 0]
            ax.plot(alphas, losses, '-', linewidth=2.5, color=color, alpha=0.8)
            ax.axhline(y=base_loss, color='red', linestyle='--', linewidth=2, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

            # Mark optimal
            min_result = analysis['min_general_loss']
            ax.scatter([min_result.alpha], [min_result.loss],
                      color=color, s=120, marker='D',
                      edgecolors='black', linewidth=1.5, zorder=5)

            # Mark zero-crossings
            if analysis['zero_crossings']:
                zc_alphas = [r.alpha for r in analysis['zero_crossings']]
                zc_losses = [r.loss for r in analysis['zero_crossings']]
                ax.scatter(zc_alphas, zc_losses, color='orange', s=120,
                          marker='*', edgecolors='black', linewidth=1, zorder=5)

            ax.set_xlabel('α', fontsize=11)
            ax.set_ylabel('Loss', fontsize=11)
            ax.set_title(f'{task_name}: Loss Landscape', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Right: Functional Return
            ax = axes[idx, 1]
            ax.plot(alphas, returns, '-', linewidth=2.5, color=color, alpha=0.8)
            ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

            # Mark zero-crossings
            if analysis['zero_crossings']:
                zc_returns = [r.functional_return for r in analysis['zero_crossings']]
                ax.scatter(zc_alphas, zc_returns, color='green', s=120,
                          marker='*', edgecolors='black', linewidth=1, zorder=5)

            ax.set_xlabel('α', fontsize=11)
            ax.set_ylabel('|L(α) - L(M_base)|', fontsize=11)
            ax.set_title(f'{task_name}: Functional Return', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n📊 Side-by-side plot saved to {output_path}")
        return output_path

    def _plot_grid_comparison(
        self,
        task_datasets: Dict[str, Dict[str, Any]],
        output_path: str,
        colors: Dict[str, str]
    ) -> str:
        """Create detailed grid comparison (3×3 grid with loss, return, squaring)."""
        num_tasks = len(task_datasets)
        fig, axes = plt.subplots(num_tasks, 3, figsize=(18, 4.5 * num_tasks))

        if num_tasks == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('Detailed Grid Comparison: Loss Landscapes with Squaring Tests\n'
                     'Inspired by Eckmann & Tlusty (2025)',
                     fontsize=14, fontweight='bold', y=0.998)

        for idx, (task_name, data) in enumerate(task_datasets.items()):
            results = data['results']
            analysis = data['analysis']

            alphas = [r.alpha for r in results]
            losses = [r.loss for r in results]
            returns = [r.functional_return for r in results]
            base_loss = results[0].base_loss if results else 0

            color = colors.get(task_name, 'gray')

            # Column 1: Loss Landscape
            ax = axes[idx, 0]
            ax.plot(alphas, losses, '-', linewidth=2.5, color=color, alpha=0.8, label='L(α)')
            ax.axhline(y=base_loss, color='red', linestyle='--', linewidth=2,
                      alpha=0.5, label='L_base')
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

            min_result = analysis['min_general_loss']
            ax.scatter([min_result.alpha], [min_result.loss],
                      color=color, s=100, marker='D',
                      edgecolors='black', linewidth=1, zorder=5, label='Optimal')

            ax.set_xlabel('α', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title(f'{task_name}: Loss Landscape', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Column 2: Functional Return
            ax = axes[idx, 1]
            ax.plot(alphas, returns, '-', linewidth=2.5, color=color, alpha=0.8)
            ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

            if analysis['zero_crossings']:
                zc_alphas = [r.alpha for r in analysis['zero_crossings']]
                zc_returns = [r.functional_return for r in analysis['zero_crossings']]
                ax.scatter(zc_alphas, zc_returns, color='green', s=100,
                          marker='*', edgecolors='black', linewidth=1, zorder=5)

            ax.set_xlabel('α', fontsize=10)
            ax.set_ylabel('|L(α) - L_base|', fontsize=10)
            ax.set_title(f'{task_name}: Functional Return', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Column 3: Squaring Test
            ax = axes[idx, 2]
            if analysis.get('has_squaring_data', False):
                losses_2alpha = [r.loss_2alpha for r in results if hasattr(r, 'loss_2alpha')]
                alphas_sq = [r.alpha for r in results if hasattr(r, 'loss_2alpha')]

                if losses_2alpha:
                    ax.plot(alphas, losses, '-', linewidth=2, color=color, alpha=0.6, label='L(α)')
                    ax.plot(alphas_sq, losses_2alpha, '--', linewidth=2,
                           color=color, alpha=0.8, label='L(2α)')
                    ax.axhline(y=base_loss, color='green', linestyle='--',
                              linewidth=2, alpha=0.5, label='L_base')

                    if 'squaring_return_points' in analysis:
                        sr_alphas = [r.alpha for r in analysis['squaring_return_points']]
                        sr_losses = [r.loss_2alpha for r in analysis['squaring_return_points']]
                        ax.scatter(sr_alphas, sr_losses, color='orange', s=100,
                                  marker='*', edgecolors='black', linewidth=1, zorder=5)

                    ax.set_xlabel('α', fontsize=10)
                    ax.set_ylabel('Loss', fontsize=10)
                    ax.set_title(f'{task_name}: Squaring Test', fontsize=11, fontweight='bold')
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No squaring data',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{task_name}: Squaring Test', fontsize=11, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No squaring data',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{task_name}: Squaring Test', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n📊 Grid plot saved to {output_path}")
        return output_path

    def _plot_publication_quality(
        self,
        task_datasets: Dict[str, Dict[str, Any]],
        output_path: str,
        colors: Dict[str, str]
    ) -> str:
        """Create publication-quality single panel with insets."""
        fig = plt.figure(figsize=(12, 8))

        # Main panel (takes most of the space)
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)

        # Inset panels
        ax_inset1 = plt.subplot2grid((3, 3), (0, 2))
        ax_inset2 = plt.subplot2grid((3, 3), (1, 2))
        ax_inset3 = plt.subplot2grid((3, 3), (2, 2))

        # Main panel: Overlaid loss curves
        for task_name, data in task_datasets.items():
            results = data['results']
            analysis = data['analysis']

            alphas = [r.alpha for r in results]
            losses = [r.loss for r in results]
            base_loss = results[0].base_loss if results else 0
            color = colors.get(task_name, 'gray')

            ax_main.plot(alphas, losses, '-', linewidth=3,
                        color=color, label=task_name, alpha=0.85)

            # Mark optimal
            min_result = analysis['min_general_loss']
            ax_main.scatter([min_result.alpha], [min_result.loss],
                           color=color, s=150, marker='D',
                           edgecolors='black', linewidth=1.5, zorder=5)

            # Annotate optimal alpha
            ax_main.annotate(f'α*={min_result.alpha:.3f}',
                            xy=(min_result.alpha, min_result.loss),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=9, color=color, weight='bold',
                            bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='white', edgecolor=color, alpha=0.8))

        # Base loss reference
        first_data = list(task_datasets.values())[0]
        base_loss = first_data['results'][0].base_loss
        ax_main.axhline(y=base_loss, color='black', linestyle='--',
                       linewidth=2, label='Base Model', alpha=0.6)
        ax_main.axvline(x=0, color='gray', linestyle=':', alpha=0.4)

        ax_main.set_xlabel('Task Vector Scaling Factor (α)', fontsize=13, weight='bold')
        ax_main.set_ylabel('Loss L(M_base + αT)', fontsize=13, weight='bold')
        ax_main.set_title('Multi-Task Loss Landscape Comparison\n'
                         'Exploring Self-Inverse Properties in Neural Network Task Vectors',
                         fontsize=14, weight='bold', pad=15)
        ax_main.legend(fontsize=11, loc='best', framealpha=0.95)
        ax_main.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)

        # Inset 1: Optimal alpha comparison
        task_names = list(task_datasets.keys())
        optimal_alphas = [task_datasets[t]['analysis']['min_general_loss'].alpha
                         for t in task_names]
        task_colors = [colors.get(t, 'gray') for t in task_names]

        ax_inset1.bar(range(len(task_names)), optimal_alphas, color=task_colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5)
        ax_inset1.set_xticks(range(len(task_names)))
        ax_inset1.set_xticklabels(task_names, fontsize=9)
        ax_inset1.set_ylabel('Optimal α', fontsize=9, weight='bold')
        ax_inset1.set_title('Optimal Scaling', fontsize=10, weight='bold')
        ax_inset1.grid(True, alpha=0.3, axis='y')
        ax_inset1.spines['top'].set_visible(False)
        ax_inset1.spines['right'].set_visible(False)

        # Inset 2: Loss improvement
        loss_improvements = [
            task_datasets[t]['analysis']['min_general_loss'].base_loss -
            task_datasets[t]['analysis']['min_general_loss'].loss
            for t in task_names
        ]

        ax_inset2.bar(range(len(task_names)), loss_improvements, color=task_colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5)
        ax_inset2.set_xticks(range(len(task_names)))
        ax_inset2.set_xticklabels(task_names, fontsize=9)
        ax_inset2.set_ylabel('Δ Loss', fontsize=9, weight='bold')
        ax_inset2.set_title('Loss Improvement', fontsize=10, weight='bold')
        ax_inset2.grid(True, alpha=0.3, axis='y')
        ax_inset2.spines['top'].set_visible(False)
        ax_inset2.spines['right'].set_visible(False)

        # Inset 3: Zero-crossing count
        zc_counts = [len(task_datasets[t]['analysis']['zero_crossings']) for t in task_names]

        ax_inset3.bar(range(len(task_names)), zc_counts, color=task_colors, alpha=0.7,
                     edgecolor='black', linewidth=1.5)
        ax_inset3.set_xticks(range(len(task_names)))
        ax_inset3.set_xticklabels(task_names, fontsize=9)
        ax_inset3.set_ylabel('Count', fontsize=9, weight='bold')
        ax_inset3.set_title('Zero-Crossings', fontsize=10, weight='bold')
        ax_inset3.grid(True, alpha=0.3, axis='y')
        ax_inset3.spines['top'].set_visible(False)
        ax_inset3.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n📊 Publication-quality plot saved to {output_path}")
        return output_path

    def _plot_heatmap_comparison(
        self,
        task_datasets: Dict[str, Dict[str, Any]],
        output_path: str,
        colors: Dict[str, str]
    ) -> str:
        """Create averaged loss landscape heatmap revealing neural network structure.

        Averages loss across all tasks at each α to show the universal
        loss landscape structure that's task-agnostic.
        """
        import numpy as np

        # Get task names in consistent order
        task_names = sorted(task_datasets.keys())

        # Find common alpha range
        all_alphas = set()
        for data in task_datasets.values():
            all_alphas.update(r.alpha for r in data['results'])
        alphas_sorted = sorted(all_alphas)

        # Build loss matrix (tasks × alphas)
        loss_matrix = np.zeros((len(task_names), len(alphas_sorted)))

        for i, task_name in enumerate(task_names):
            results = task_datasets[task_name]['results']
            alpha_to_loss = {r.alpha: r.loss for r in results}

            for j, alpha in enumerate(alphas_sorted):
                loss_matrix[i, j] = alpha_to_loss.get(alpha, np.nan)

        # Compute aggregated statistics across tasks
        mean_loss = np.nanmean(loss_matrix, axis=0)
        std_loss = np.nanstd(loss_matrix, axis=0)
        min_loss_per_alpha = np.nanmin(loss_matrix, axis=0)
        max_loss_per_alpha = np.nanmax(loss_matrix, axis=0)

        # Create 2×2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        fig.suptitle('Averaged Loss Landscape: Universal Neural Network Structure\n'
                     'Aggregated Across All Tasks',
                     fontsize=14, fontweight='bold', y=0.995)

        # ===== Panel 1: Mean loss curve (main result) =====
        ax = axes[0, 0]
        ax.plot(alphas_sorted, mean_loss, 'b-', linewidth=3, label='Mean Loss', zorder=3)
        ax.fill_between(alphas_sorted, mean_loss - std_loss, mean_loss + std_loss,
                        alpha=0.3, color='blue', label='±1 Std Dev')

        # Mark global minimum
        min_idx = np.nanargmin(mean_loss)
        optimal_alpha = alphas_sorted[min_idx]
        ax.plot(optimal_alpha, mean_loss[min_idx], 'r*', markersize=20,
                markeredgewidth=2, markeredgecolor='white', zorder=5,
                label=f'Optimal α={optimal_alpha:.3f}')

        # Mark base model location
        base_idx = min(range(len(alphas_sorted)), key=lambda i: abs(alphas_sorted[i]))
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=2,
                  label='Base Model (α=0)')

        ax.set_xlabel('Task Vector Scaling Factor (α)', fontsize=12, weight='bold')
        ax.set_ylabel('Mean Loss (Averaged Across Tasks)', fontsize=12, weight='bold')
        ax.set_title('Universal Loss Landscape Structure', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # ===== Panel 2: Variance analysis =====
        ax = axes[0, 1]
        ax.plot(alphas_sorted, std_loss, 'r-', linewidth=2.5, label='Std Dev')
        ax.fill_between(alphas_sorted, 0, std_loss, alpha=0.3, color='red')
        ax.axvline(x=optimal_alpha, color='green', linestyle='--', alpha=0.5,
                  label=f'Optimal α={optimal_alpha:.3f}')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Task Vector Scaling Factor (α)', fontsize=12, weight='bold')
        ax.set_ylabel('Standard Deviation', fontsize=12, weight='bold')
        ax.set_title('Task Variance (Lower = More Universal)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # ===== Panel 3: Individual task curves overlaid on mean =====
        ax = axes[1, 0]

        # Plot mean as thick line
        ax.plot(alphas_sorted, mean_loss, 'k-', linewidth=4, label='Mean', alpha=0.8, zorder=5)

        # Plot individual tasks
        for i, task_name in enumerate(task_names):
            task_color = colors.get(task_name, 'gray')
            ax.plot(alphas_sorted, loss_matrix[i, :], '-', linewidth=2,
                   color=task_color, alpha=0.6, label=task_name, zorder=3)

        ax.plot(optimal_alpha, mean_loss[min_idx], 'r*', markersize=20,
                markeredgewidth=2, markeredgecolor='white', zorder=10)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Task Vector Scaling Factor (α)', fontsize=12, weight='bold')
        ax.set_ylabel('Loss', fontsize=12, weight='bold')
        ax.set_title('Individual Tasks vs Mean', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # ===== Panel 4: Heatmap showing all tasks + mean =====
        ax = axes[1, 1]

        # Add mean as extra row to matrix
        display_matrix = np.vstack([loss_matrix, mean_loss.reshape(1, -1)])
        display_labels = task_names + ['MEAN']

        # Sample every Nth alpha for readability
        sample_step = max(1, len(alphas_sorted) // 50)
        sampled_indices = np.arange(0, len(alphas_sorted), sample_step)
        sampled_matrix = display_matrix[:, sampled_indices]
        sampled_alphas = [alphas_sorted[i] for i in sampled_indices]

        im = ax.imshow(sampled_matrix, aspect='auto', cmap='viridis', interpolation='bilinear')

        # Set ticks
        ax.set_yticks(range(len(display_labels)))
        ax.set_yticklabels(display_labels, fontsize=11)
        # Bold the MEAN row
        ytick_labels = ax.get_yticklabels()
        ytick_labels[-1].set_weight('bold')
        ytick_labels[-1].set_fontsize(12)

        # X-axis ticks
        tick_step = max(1, len(sampled_alphas) // 10)
        tick_indices = np.arange(0, len(sampled_alphas), tick_step)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f'{sampled_alphas[i]:.2f}' for i in tick_indices],
                          fontsize=9, rotation=45)

        # Mark optimal on mean row
        optimal_col = min(range(len(sampled_alphas)),
                         key=lambda i: abs(sampled_alphas[i] - optimal_alpha))
        ax.plot(optimal_col, len(task_names), 'r*', markersize=15,
                markeredgewidth=2, markeredgecolor='white', zorder=10)

        ax.set_xlabel('Task Vector Scaling Factor (α)', fontsize=11, weight='bold')
        ax.set_title('Heatmap: Tasks + Averaged Structure', fontsize=13, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Loss', fontsize=10, weight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Print key findings
        print(f"\n📊 Heatmap plot saved to {output_path}")
        print(f"\n   Universal optimal α: {optimal_alpha:.4f}")
        print(f"   Mean loss at optimal: {mean_loss[min_idx]:.4f}")
        print(f"   Std dev at optimal: {std_loss[min_idx]:.4f}")

        return output_path
