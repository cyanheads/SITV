"""
2D Composition Analysis Module

Analyzes the relationship between individual task vector properties
and optimal composition in 2D task vector space.

This module investigates whether we can predict optimal (α, β) from
individual task vector properties like magnitude, norms, and 1D optima.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator  # type: ignore[import-untyped]
from typing import Dict, Any, Tuple, List, Optional


class CompositionAnalyzer:
    """Analyzes 2D task vector composition landscapes.

    This analyzer explores the interaction between two task vectors,
    testing whether optimal 2D combinations can be predicted from
    individual 1D properties.

    Attributes:
        data_1d: 1D alpha sweep results
        data_2d: 2D composition grid results
        analysis: Analysis metadata and properties
    """

    def __init__(self, data_dir: Path):
        """Initialize analyzer with data from a directory.

        Args:
            data_dir: Directory containing JSON result files
        """
        self.data_dir = Path(data_dir)
        self.data_1d, self.data_2d, self.analysis = self._load_data()

    def _load_data(self) -> Tuple[List[Dict], List[Dict], Dict]:
        """Load 1D and 2D composition data from JSON files."""
        # Load 1D T1 data
        results_1d_path = self.data_dir / "loss_landscape_results.json"
        results_2d_path = self.data_dir / "loss_landscape_2d_results.json"
        analysis_path = self.data_dir / "analysis_results.json"

        if not results_1d_path.exists():
            raise FileNotFoundError(f"1D results not found: {results_1d_path}")
        if not results_2d_path.exists():
            raise FileNotFoundError(f"2D results not found: {results_2d_path}")
        if not analysis_path.exists():
            raise FileNotFoundError(f"Analysis results not found: {analysis_path}")

        with open(results_1d_path) as f:
            data_1d = json.load(f)

        with open(results_2d_path) as f:
            data_2d = json.load(f)

        with open(analysis_path) as f:
            analysis = json.load(f)

        return data_1d, data_2d, analysis

    def extract_1d_properties(self) -> Dict[str, Any]:
        """Extract key properties from 1D sweep.

        Returns:
            Dictionary containing optimal alpha, loss, curvature, etc.
        """
        alphas = np.array([p["alpha"] for p in self.data_1d])
        losses = np.array([p["loss"] for p in self.data_1d])

        # Find optimal
        opt_idx = np.argmin(losses)
        alpha_opt = alphas[opt_idx]
        loss_opt = losses[opt_idx]

        # Estimate curvature at optimal point (second derivative)
        if opt_idx > 0 and opt_idx < len(alphas) - 1:
            h = alphas[1] - alphas[0]
            d2L = (losses[opt_idx + 1] - 2 * losses[opt_idx] + losses[opt_idx - 1]) / (h ** 2)
        else:
            d2L = None

        # Find zero-crossings (where loss ≈ base_loss)
        base_loss = self.data_1d[0]["base_loss"]
        zero_crossings = []

        for i in range(len(alphas) - 1):
            if (losses[i] - base_loss) * (losses[i + 1] - base_loss) < 0:
                # Linear interpolation
                alpha_cross = alphas[i] + (base_loss - losses[i]) / (
                    losses[i + 1] - losses[i]
                ) * (alphas[i + 1] - alphas[i])
                zero_crossings.append(alpha_cross)

        return {
            "alpha_opt": alpha_opt,
            "loss_opt": loss_opt,
            "curvature": d2L,
            "zero_crossings": zero_crossings,
            "base_loss": base_loss,
            "alphas": alphas,
            "losses": losses,
        }

    def extract_2d_properties(self) -> Dict[str, Any]:
        """Extract key properties from 2D composition.

        Returns:
            Dictionary containing optimal (alpha, beta), loss grid, etc.
        """
        points = [(p["alpha"], p["beta"]) for p in self.data_2d]
        losses = np.array([p["loss"] for p in self.data_2d])

        # Find optimal
        opt_idx = np.argmin(losses)
        alpha_opt, beta_opt = points[opt_idx]
        loss_opt = losses[opt_idx]

        # Reconstruct grid
        alphas = sorted(set(p["alpha"] for p in self.data_2d))
        betas = sorted(set(p["beta"] for p in self.data_2d))

        # Create loss grid
        loss_grid = np.zeros((len(alphas), len(betas)))
        for p in self.data_2d:
            i = alphas.index(p["alpha"])
            j = betas.index(p["beta"])
            loss_grid[i, j] = p["loss"]

        return {
            "alpha_opt": alpha_opt,
            "beta_opt": beta_opt,
            "loss_opt": loss_opt,
            "base_loss": self.data_2d[0]["base_loss"],
            "alphas": np.array(alphas),
            "betas": np.array(betas),
            "loss_grid": loss_grid,
        }

    def analyze_cross_sections(self, props_2d: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Analyze 1D cross-sections through the 2D landscape.

        Args:
            props_2d: 2D properties from extract_2d_properties()

        Returns:
            Dictionary of cross-section data
        """
        alphas = props_2d["alphas"]
        betas = props_2d["betas"]
        loss_grid = props_2d["loss_grid"]

        # Create interpolator
        interp = RegularGridInterpolator((alphas, betas), loss_grid, method='linear')

        alpha_opt = props_2d["alpha_opt"]
        beta_opt = props_2d["beta_opt"]

        # Cross-section 1: Fix β at optimal, vary α
        alpha_range = np.linspace(alphas[0], alphas[-1], 200)
        loss_at_beta_opt = [interp([a, beta_opt])[0] for a in alpha_range]

        # Cross-section 2: Fix α at optimal, vary β
        beta_range = np.linspace(betas[0], betas[-1], 200)
        loss_at_alpha_opt = [interp([alpha_opt, b])[0] for b in beta_range]

        # Cross-section 3: β = 0 (no T2)
        if 0 in betas:
            loss_at_beta_0 = loss_grid[:, list(betas).index(0)]
        else:
            loss_at_beta_0 = [interp([a, 0])[0] for a in alphas]

        # Cross-section 4: α = 0 (no T1)
        if 0 in alphas:
            loss_at_alpha_0 = loss_grid[list(alphas).index(0), :]
        else:
            loss_at_alpha_0 = [interp([0, b])[0] for b in betas]

        return {
            "alpha_range": alpha_range,
            "loss_at_beta_opt": np.array(loss_at_beta_opt),
            "beta_range": beta_range,
            "loss_at_alpha_opt": np.array(loss_at_alpha_opt),
            "loss_at_beta_0": np.array(loss_at_beta_0),
            "loss_at_alpha_0": np.array(loss_at_alpha_0),
        }

    def estimate_interaction_strength(
        self,
        props_2d: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate interaction strength between T1 and T2.

        If tasks are independent: L(α, β) ≈ L(α, 0) + L(0, β) - L(0, 0)
        Deviation from this indicates interaction.

        Args:
            props_2d: 2D properties from extract_2d_properties()

        Returns:
            Dictionary with interaction metrics and R²
        """
        alphas = props_2d["alphas"]
        betas = props_2d["betas"]
        loss_grid = props_2d["loss_grid"]

        # Get L(0, 0)
        idx_alpha_0 = np.argmin(np.abs(alphas))
        idx_beta_0 = np.argmin(np.abs(betas))
        L_00 = loss_grid[idx_alpha_0, idx_beta_0]

        # Additive model: L_additive(α, β) = L(α, 0) + L(0, β) - L(0, 0)
        loss_additive = np.zeros_like(loss_grid)
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                L_alpha_0 = loss_grid[i, idx_beta_0]
                L_0_beta = loss_grid[idx_alpha_0, j]
                loss_additive[i, j] = L_alpha_0 + L_0_beta - L_00

        # Interaction term
        interaction = loss_grid - loss_additive

        # Measure strength
        interaction_rms = np.sqrt(np.mean(interaction ** 2))

        # R² of additive model
        total_var = np.var(loss_grid)
        residual_var = np.var(interaction)
        r_squared = 1 - residual_var / total_var

        return {
            "loss_additive": loss_additive,
            "interaction": interaction,
            "interaction_rms": interaction_rms,
            "r_squared": r_squared,
        }

    def predict_optimal_from_1d(
        self,
        props_1d: Dict[str, Any],
        props_1d_t2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Tuple[float, float]]:
        """Predict 2D optimal from 1D properties.

        Tests simple heuristic models:
        1. Independent: (α*, β*) = (α1*, α2*)
        2. Opposing: (α*, β*) = (α1*, -α1*)
        3. Conservative: Reduce both by half

        Args:
            props_1d: 1D properties for first task vector
            props_1d_t2: Optional 1D properties for second task vector

        Returns:
            Dictionary of prediction strategies and their (α, β) predictions
        """
        alpha1_opt = props_1d["alpha_opt"]

        if props_1d_t2 is None:
            predictions = {
                "independent": (alpha1_opt, alpha1_opt),
                "opposing": (alpha1_opt, -alpha1_opt),
                "conservative": (alpha1_opt / 2, -alpha1_opt / 2),
            }
        else:
            alpha2_opt = props_1d_t2["alpha_opt"]
            predictions = {
                "independent": (alpha1_opt, alpha2_opt),
                "symmetric": (alpha1_opt, -alpha1_opt),
            }

        return predictions

    def visualize(
        self,
        props_1d: Dict[str, Any],
        props_2d: Dict[str, Any],
        cross_sections: Dict[str, np.ndarray],
        interaction_analysis: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Path:
        """Create comprehensive visualization of the analysis.

        Args:
            props_1d: 1D properties
            props_2d: 2D properties
            cross_sections: Cross-section data
            interaction_analysis: Interaction metrics
            output_path: Where to save (default: data_dir/composition_analysis.png)

        Returns:
            Path to saved visualization
        """
        if output_path is None:
            output_path = self.data_dir / "composition_analysis.png"

        fig = plt.figure(figsize=(16, 12))

        # 1. 2D heatmap with optimal point
        ax1 = plt.subplot(2, 3, 1)
        alphas = props_2d["alphas"]
        betas = props_2d["betas"]
        extent = (float(betas[0]), float(betas[-1]), float(alphas[0]), float(alphas[-1]))
        im1 = ax1.imshow(props_2d["loss_grid"], origin='lower', extent=extent,
                         aspect='auto', cmap='viridis', vmin=0, vmax=20)
        ax1.plot(props_2d["beta_opt"], props_2d["alpha_opt"], 'r*', markersize=15,
                 label=f'Optimal ({props_2d["alpha_opt"]:.2f}, {props_2d["beta_opt"]:.2f})')
        ax1.axhline(0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax1.axvline(0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax1.set_xlabel('β (T2 scaling)')
        ax1.set_ylabel('α (T1 scaling)')
        ax1.set_title('2D Loss Landscape L(α, β)')
        ax1.legend()
        plt.colorbar(im1, ax=ax1, label='Loss')

        # 2. Cross-section at optimal β
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(cross_sections["alpha_range"], cross_sections["loss_at_beta_opt"],
                 'b-', label=f'L(α, β={props_2d["beta_opt"]:.2f})')
        ax2.plot(props_1d["alphas"], props_1d["losses"], 'g--',
                 label='L(α, 0) [1D T1 only]', alpha=0.7)
        ax2.axvline(props_2d["alpha_opt"], color='red', linestyle='--',
                    label=f'2D optimal α={props_2d["alpha_opt"]:.2f}')
        ax2.axvline(props_1d["alpha_opt"], color='green', linestyle=':',
                    label=f'1D optimal α={props_1d["alpha_opt"]:.2f}')
        ax2.axhline(props_2d["base_loss"], color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('α')
        ax2.set_ylabel('Loss')
        ax2.set_title('Cross-section: Fix β at Optimal')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Cross-section at optimal α
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(cross_sections["beta_range"], cross_sections["loss_at_alpha_opt"],
                 'r-', label=f'L(α={props_2d["alpha_opt"]:.2f}, β)')
        ax3.axvline(props_2d["beta_opt"], color='red', linestyle='--',
                    label=f'2D optimal β={props_2d["beta_opt"]:.2f}')
        ax3.axhline(props_2d["base_loss"], color='gray', linestyle=':', alpha=0.5)
        ax3.set_xlabel('β')
        ax3.set_ylabel('Loss')
        ax3.set_title('Cross-section: Fix α at Optimal')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Interaction heatmap
        ax4 = plt.subplot(2, 3, 4)
        extent_tuple = (float(betas[0]), float(betas[-1]), float(alphas[0]), float(alphas[-1]))
        im4 = ax4.imshow(interaction_analysis["interaction"], origin='lower',
                         extent=extent_tuple, aspect='auto', cmap='RdBu_r')
        ax4.plot(props_2d["beta_opt"], props_2d["alpha_opt"], 'k*', markersize=15)
        ax4.axhline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax4.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax4.set_xlabel('β (T2 scaling)')
        ax4.set_ylabel('α (T1 scaling)')
        ax4.set_title(f'Interaction: L_actual - L_additive\nR²={interaction_analysis["r_squared"]:.3f}')
        plt.colorbar(im4, ax=ax4, label='Interaction')

        # 5. Additive model prediction
        ax5 = plt.subplot(2, 3, 5)
        im5 = ax5.imshow(interaction_analysis["loss_additive"], origin='lower',
                         extent=extent_tuple, aspect='auto', cmap='viridis', vmin=0, vmax=20)
        ax5.plot(props_2d["beta_opt"], props_2d["alpha_opt"], 'r*', markersize=15)
        ax5.axhline(0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax5.axvline(0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax5.set_xlabel('β (T2 scaling)')
        ax5.set_ylabel('α (T1 scaling)')
        ax5.set_title('Additive Model: L(α,0) + L(0,β) - L(0,0)')
        plt.colorbar(im5, ax=ax5, label='Loss')

        # 6. Comparison: axes cross-sections
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(alphas, cross_sections["loss_at_beta_0"], 'g-',
                 label='L(α, 0) [no T2]', linewidth=2)
        ax6.plot(betas, cross_sections["loss_at_alpha_0"], 'b-',
                 label='L(0, β) [no T1]', linewidth=2)
        ax6.axvline(props_2d["alpha_opt"], color='green', linestyle='--', alpha=0.5)
        ax6.axvline(props_2d["beta_opt"], color='blue', linestyle='--', alpha=0.5)
        ax6.axhline(props_2d["base_loss"], color='gray', linestyle=':', alpha=0.5)
        ax6.set_xlabel('α or β')
        ax6.set_ylabel('Loss')
        ax6.set_title('Individual Task Contributions')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def run_full_analysis(
        self,
        save_results: bool = True,
        save_visualization: bool = True
    ) -> Dict[str, Any]:
        """Run complete composition analysis pipeline.

        Args:
            save_results: Whether to save JSON results
            save_visualization: Whether to save plots

        Returns:
            Dictionary containing all analysis results
        """
        print("=" * 80)
        print("ANALYSIS: Predicting Optimal Composition from Individual Task Properties")
        print("=" * 80)
        print()

        # Extract properties
        print("Extracting properties...")
        props_1d = self.extract_1d_properties()
        props_2d = self.extract_2d_properties()

        print("\n" + "=" * 80)
        print("1D TASK VECTOR")
        print("=" * 80)
        print(f"Optimal α: {props_1d['alpha_opt']:.4f}")
        print(f"Loss at optimal: {props_1d['loss_opt']:.4f}")
        print(f"Base loss: {props_1d['base_loss']:.4f}")
        if props_1d['curvature']:
            print(f"Curvature at optimal: {props_1d['curvature']:.4f}")
        print(f"Zero-crossings: {[f'{z:.4f}' for z in props_1d['zero_crossings']]}")

        print("\n" + "=" * 80)
        print("2D COMPOSITION")
        print("=" * 80)
        print(f"Optimal (α, β): ({props_2d['alpha_opt']:.4f}, {props_2d['beta_opt']:.4f})")
        print(f"Loss at optimal: {props_2d['loss_opt']:.4f}")
        print(f"Base loss: {props_2d['base_loss']:.4f}")

        # Analyze cross-sections
        print("\nAnalyzing cross-sections...")
        cross_sections = self.analyze_cross_sections(props_2d)

        # Estimate interaction
        print("Estimating interaction strength...")
        interaction_analysis = self.estimate_interaction_strength(props_2d)

        print("\n" + "=" * 80)
        print("INTERACTION ANALYSIS")
        print("=" * 80)
        print(f"Interaction RMS: {interaction_analysis['interaction_rms']:.4f}")
        print(f"Additive model R²: {interaction_analysis['r_squared']:.4f}")
        print()
        if interaction_analysis['r_squared'] > 0.8:
            print("✓ Tasks are mostly independent (additive model fits well)")
        elif interaction_analysis['r_squared'] > 0.5:
            print("⚠ Moderate interaction between tasks")
        else:
            print("✗ Strong interaction between tasks (non-additive)")

        # Make predictions
        print("\n" + "=" * 80)
        print("PREDICTIONS FROM 1D PROPERTIES")
        print("=" * 80)
        predictions = self.predict_optimal_from_1d(props_1d)
        actual = (props_2d['alpha_opt'], props_2d['beta_opt'])

        print(f"Actual 2D optimal: {actual}")
        print("\nPrediction strategies:")
        prediction_errors = {}
        for name, pred in predictions.items():
            error = np.sqrt((pred[0] - actual[0])**2 + (pred[1] - actual[1])**2)
            prediction_errors[name] = error
            print(f"  {name:20s}: {pred} → error = {error:.4f}")

        # Visualize
        if save_visualization:
            print("\nGenerating visualization...")
            viz_path = self.visualize(props_1d, props_2d, cross_sections, interaction_analysis)
            print(f"Saved visualization to {viz_path}")

        # Prepare results
        results = {
            "1d_properties": {
                "alpha_opt": float(props_1d["alpha_opt"]),
                "loss_opt": float(props_1d["loss_opt"]),
                "curvature": float(props_1d["curvature"]) if props_1d["curvature"] else None,
                "zero_crossings": [float(z) for z in props_1d["zero_crossings"]],
            },
            "2d_properties": {
                "alpha_opt": float(props_2d["alpha_opt"]),
                "beta_opt": float(props_2d["beta_opt"]),
                "loss_opt": float(props_2d["loss_opt"]),
            },
            "interaction": {
                "rms": float(interaction_analysis["interaction_rms"]),
                "r_squared": float(interaction_analysis["r_squared"]),
            },
            "predictions": {k: (float(v[0]), float(v[1])) for k, v in predictions.items()},
            "prediction_errors": {k: float(v) for k, v in prediction_errors.items()}
        }

        # Save results
        if save_results:
            output_path = self.data_dir / "composition_prediction_analysis.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved analysis to {output_path}")

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)

        return results


def main():
    """Command-line interface for composition analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze 2D task vector composition and predict optimal combinations"
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory containing loss_landscape_results.json and loss_landscape_2d_results.json'
    )
    parser.add_argument(
        '--no-save-results',
        action='store_true',
        help='Skip saving JSON results'
    )
    parser.add_argument(
        '--no-save-viz',
        action='store_true',
        help='Skip saving visualization'
    )

    args = parser.parse_args()

    # Run analysis
    analyzer = CompositionAnalyzer(Path(args.data_dir))
    analyzer.run_full_analysis(
        save_results=not args.no_save_results,
        save_visualization=not args.no_save_viz
    )


if __name__ == "__main__":
    main()
