"""
2D Composition Analysis Module

Analyzes the relationship between individual task vector properties
and optimal composition in 2D task vector space.

This module investigates whether we can predict optimal (α, β) from
individual task vector properties like magnitude, norms, and 1D optima.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator  # type: ignore[import-untyped]

# --------------------------- helpers ---------------------------

def _closest_index_to_zero(arr: np.ndarray) -> int:
    """Index of the element in arr closest to 0."""
    return int(np.nanargmin(np.abs(arr)))


def _robust_unique_sorted(vals: np.ndarray, decimals: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return unique sorted values (rounded) and inverse indices.

    Rounding tames tiny float jitter from grid generation.
    """
    vals = np.asarray(vals, dtype=float)
    rounded = np.round(vals, decimals=decimals)
    uniq, inv = np.unique(rounded, return_inverse=True)
    return uniq, inv


def _polyfit_curvature(x: np.ndarray, y: np.ndarray, k: int = 1) -> float | None:
    """Estimate local curvature (2nd deriv) by quadratic fit around argmin.

    Picks 2k+1 points centered on the min if possible; falls back to None.
    """
    if len(x) < 3 or len(x) != len(y):
        return None
    # sort by x
    order = np.argsort(x)
    x, y = x[order], y[order]
    i = int(np.argmin(y))
    left = max(0, i - k)
    right = min(len(x), i + k + 1)
    if right - left < 3:
        # widen window if we hit boundaries
        left = max(0, i - 1)
        right = min(len(x), i + 2)
    xs, ys = x[left:right], y[left:right]
    if len(xs) < 3:
        return None
    try:
        coeffs = np.polyfit(xs, ys, 2)  # ax^2 + bx + c
        a = float(coeffs[0])
        return 2.0 * a
    except Exception:
        return None


# --------------------------- analyzer ---------------------------

class CompositionAnalyzer:
    """Analyzes 2D task vector composition landscapes.

    This analyzer explores the interaction between two task vectors,
    testing whether optimal 2D combinations can be predicted from
    individual 1D properties.

    Attributes:
        data_1d: 1D alpha sweep results
        data_2d: 2D composition grid results
        analysis: Analysis metadata and properties (optional)
    """

    def __init__(self, data_dir: Path, t2_1d_path: Path | None = None):
        """Initialize analyzer with data from a directory.

        Args:
            data_dir: Directory containing JSON result files
            t2_1d_path: Optional path to T2's 1D alpha sweep results (for improved predictions)
        """
        self.data_dir = Path(data_dir)
        self.data_1d, self.data_2d, self.analysis = self._load_data()
        self.data_1d_t2 = self._load_t2_1d(t2_1d_path)

    def _load_data(self) -> tuple[list[dict], list[dict], dict]:
        """Load 1D and 2D composition data from JSON files."""
        results_1d_path = self.data_dir / "loss_landscape_results.json"
        results_2d_path = self.data_dir / "loss_landscape_2d_results.json"
        analysis_path = self.data_dir / "analysis_results.json"

        if not results_1d_path.exists():
            raise FileNotFoundError(f"1D results not found: {results_1d_path}")
        if not results_2d_path.exists():
            raise FileNotFoundError(f"2D results not found: {results_2d_path}")

        with open(results_1d_path) as f:
            data_1d = json.load(f)

        with open(results_2d_path) as f:
            data_2d = json.load(f)

        analysis: dict[str, Any] = {}
        if analysis_path.exists():
            try:
                with open(analysis_path) as f:
                    analysis = json.load(f)
            except Exception:
                # Optional file; ignore parse errors.
                analysis = {}

        return data_1d, data_2d, analysis

    def _load_t2_1d(self, t2_1d_path: Path | None) -> list[dict] | None:
        """Load T2's 1D alpha sweep results if available.

        Args:
            t2_1d_path: Optional explicit path to T2's 1D results

        Returns:
            List of T2 1D sweep data points, or None if not available
        """
        # If explicit path provided, use it
        if t2_1d_path is not None:
            p = Path(t2_1d_path)
            if not p.exists():
                raise FileNotFoundError(f"T2 1D results not found: {p}")
            with open(p) as f:
                return json.load(f)  # type: ignore[no-any-return]

        # Otherwise, try auto-detecting conventional filename in data_dir
        candidate = self.data_dir / "loss_landscape_results_t2.json"
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)  # type: ignore[no-any-return]

        # Not provided and not found - return None (fallback to symmetric prediction)
        return None

    # --------------------------- 1D ---------------------------

    def extract_1d_properties(self, data_1d: list[dict] | None = None) -> dict[str, Any]:
        """Extract key properties from 1D sweep.

        Args:
            data_1d: Optional 1D sweep data. If None, uses self.data_1d

        Returns:
            Dictionary containing optimal alpha, loss, curvature, zero-crossings, etc.
        """
        # Use provided data or fall back to instance data
        data = self.data_1d if data_1d is None else data_1d
        alphas = np.asarray([p["alpha"] for p in data], dtype=float)
        losses = np.asarray([p["loss"] for p in data], dtype=float)

        # Sort by alpha for consistency
        order = np.argsort(alphas)
        alphas = alphas[order]
        losses = losses[order]

        # Optimal
        opt_idx = int(np.argmin(losses))
        alpha_opt = float(alphas[opt_idx])
        loss_opt = float(losses[opt_idx])

        # Curvature via local quadratic fit (robust to non-uniform spacing)
        curvature = _polyfit_curvature(alphas, losses, k=1)

        # Base loss as loss at alpha≈0 (not "first item")
        idx0 = _closest_index_to_zero(alphas)
        base_loss = float(losses[idx0])

        # Zero-crossings where L(α) - L(0) changes sign (linear interp)
        diffs = losses - base_loss
        zero_crossings: list[float] = []
        for i in range(len(alphas) - 1):
            y1, y2 = diffs[i], diffs[i + 1]
            if not np.isfinite(y1) or not np.isfinite(y2):
                continue
            if y1 == 0.0:
                zero_crossings.append(float(alphas[i]))
            if y1 * y2 < 0.0:
                t = -y1 / (y2 - y1)  # fraction between i and i+1
                a = float(alphas[i] + t * (alphas[i + 1] - alphas[i]))
                zero_crossings.append(a)

        return {
            "alpha_opt": alpha_opt,
            "loss_opt": loss_opt,
            "curvature": float(curvature) if curvature is not None else None,
            "zero_crossings": zero_crossings,
            "base_loss": base_loss,
            "alphas": alphas,
            "losses": losses,
        }

    # --------------------------- 2D ---------------------------

    def extract_2d_properties(self, decimals: int = 8) -> dict[str, Any]:
        """Extract key properties from 2D composition.

        Returns:
            Dictionary containing optimal (alpha, beta), loss grid, etc.
        """
        alphas_raw = np.asarray([p["alpha"] for p in self.data_2d], dtype=float)
        betas_raw = np.asarray([p["beta"] for p in self.data_2d], dtype=float)
        losses_raw = np.asarray([p["loss"] for p in self.data_2d], dtype=float)

        # Reconstruct grid indices robustly
        alphas, alpha_inv = _robust_unique_sorted(alphas_raw, decimals=decimals)
        betas, beta_inv = _robust_unique_sorted(betas_raw, decimals=decimals)

        loss_grid = np.full((len(alphas), len(betas)), np.nan, dtype=float)
        loss_grid[alpha_inv, beta_inv] = losses_raw

        # Optimal over available cells
        opt_idx_flat = int(np.nanargmin(loss_grid))
        i_opt, j_opt = np.unravel_index(opt_idx_flat, loss_grid.shape)
        alpha_opt, beta_opt = float(alphas[i_opt]), float(betas[j_opt])
        loss_opt = float(loss_grid[i_opt, j_opt])

        # Base loss L(0,0) from closest indices; fall back if NaN
        ai0 = _closest_index_to_zero(alphas)
        bj0 = _closest_index_to_zero(betas)
        base_00 = loss_grid[ai0, bj0]
        if not np.isfinite(base_00):
            # Fall back to a provided base_loss if available in any record
            fallback = next(
                (p.get("base_loss") for p in self.data_2d if "base_loss" in p),
                None,
            )
            base_00 = float(fallback) if fallback is not None else np.nan
        base_loss = float(base_00) if np.isfinite(base_00) else float("nan")

        return {
            "alpha_opt": alpha_opt,
            "beta_opt": beta_opt,
            "loss_opt": loss_opt,
            "base_loss": base_loss,
            "alphas": alphas,
            "betas": betas,
            "loss_grid": loss_grid,
        }

    # --------------------------- cross-sections ---------------------------

    def analyze_cross_sections(self, props_2d: dict[str, Any]) -> dict[str, np.ndarray]:
        """Analyze 1D cross-sections through the 2D landscape.

        Args:
            props_2d: 2D properties from extract_2d_properties()

        Returns:
            Dictionary of cross-section data
        """
        alphas = np.asarray(props_2d["alphas"], dtype=float)
        betas = np.asarray(props_2d["betas"], dtype=float)
        loss_grid = np.asarray(props_2d["loss_grid"], dtype=float)

        # Create interpolator (NaN-aware; returns NaN outside/at missing cells)
        interp = RegularGridInterpolator(
            (alphas, betas),
            loss_grid,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        alpha_opt = float(props_2d["alpha_opt"])
        beta_opt = float(props_2d["beta_opt"])

        # Cross-section 1: Fix β at optimal, vary α
        alpha_range = np.linspace(alphas[0], alphas[-1], 200)
        pts = np.column_stack([alpha_range, np.full_like(alpha_range, beta_opt)])
        loss_at_beta_opt = interp(pts)

        # Cross-section 2: Fix α at optimal, vary β
        beta_range = np.linspace(betas[0], betas[-1], 200)
        pts = np.column_stack([np.full_like(beta_range, alpha_opt), beta_range])
        loss_at_alpha_opt = interp(pts)

        # Cross-section 3: β ≈ 0 (no T2)
        idx_beta0 = np.where(np.isclose(betas, 0.0))[0]
        if idx_beta0.size:
            loss_at_beta_0 = loss_grid[:, idx_beta0[0]]
        else:
            pts = np.column_stack([alphas, np.zeros_like(alphas)])
            loss_at_beta_0 = interp(pts)

        # Cross-section 4: α ≈ 0 (no T1)
        idx_alpha0 = np.where(np.isclose(alphas, 0.0))[0]
        if idx_alpha0.size:
            loss_at_alpha_0 = loss_grid[idx_alpha0[0], :]
        else:
            pts = np.column_stack([np.zeros_like(betas), betas])
            loss_at_alpha_0 = interp(pts)

        return {
            "alpha_range": alpha_range,
            "loss_at_beta_opt": np.asarray(loss_at_beta_opt, dtype=float),
            "beta_range": beta_range,
            "loss_at_alpha_opt": np.asarray(loss_at_alpha_opt, dtype=float),
            "loss_at_beta_0": np.asarray(loss_at_beta_0, dtype=float),
            "loss_at_alpha_0": np.asarray(loss_at_alpha_0, dtype=float),
        }

    # --------------------------- interaction ---------------------------

    def estimate_interaction_strength(self, props_2d: dict[str, Any]) -> dict[str, Any]:
        """Estimate interaction strength between T1 and T2.

        If tasks are independent: L(α, β) ≈ L(α, 0) + L(0, β) - L(0, 0)
        Deviation from this indicates interaction.
        """
        alphas = np.asarray(props_2d["alphas"], dtype=float)
        betas = np.asarray(props_2d["betas"], dtype=float)
        loss_grid = np.asarray(props_2d["loss_grid"], dtype=float)

        # L(0, 0)
        ia0 = _closest_index_to_zero(alphas)
        jb0 = _closest_index_to_zero(betas)
        L_00 = loss_grid[ia0, jb0]  # noqa: N806 - Mathematical notation

        # Additive model via broadcasting
        L_alpha_0 = loss_grid[:, [jb0]]  # (A,1)  # noqa: N806 - Mathematical notation
        L_0_beta = loss_grid[[ia0], :]   # (1,B)  # noqa: N806 - Mathematical notation
        loss_additive = L_alpha_0 + L_0_beta - L_00  # (A,B)

        # Interaction term
        interaction = loss_grid - loss_additive

        # Strength (RMS) and R² (NaN-aware)
        interaction_rms = float(np.sqrt(np.nanmean(interaction ** 2)))
        total_var = float(np.nanvar(loss_grid))
        residual_var = float(np.nanvar(interaction))
        r_squared = float(1.0 - residual_var / total_var) if total_var > 0 else float("nan")

        return {
            "loss_additive": loss_additive,
            "interaction": interaction,
            "interaction_rms": interaction_rms,
            "r_squared": r_squared,
        }

    # --------------------------- predictions ---------------------------

    def predict_optimal_from_1d(
        self,
        props_1d: dict[str, Any],
        props_1d_t2: dict[str, Any] | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Predict 2D optimal from 1D properties.

        Tests simple heuristic models:
        1. Independent: (α*, β*) = (α1*, α2*) or (α1*, α1*) if α2* unknown
        2. Symmetric/opposing: (α*, β*) = (α1*, -α1*)
        3. Conservative: halve magnitudes
        """
        a1 = float(props_1d["alpha_opt"])
        if props_1d_t2 is None:
            predictions = {
                "independent": (a1, a1),
                "symmetric": (a1, -a1),
                "conservative": (a1 / 2.0, -a1 / 2.0),
            }
        else:
            a2 = float(props_1d_t2["alpha_opt"])
            predictions = {
                "independent": (a1, a2),
                "symmetric": (a1, -a1),
                "conservative": (a1 / 2.0, a2 / 2.0),
            }
        return predictions

    # --------------------------- visualization ---------------------------

    def visualize(
        self,
        props_1d: dict[str, Any],
        props_2d: dict[str, Any],
        cross_sections: dict[str, np.ndarray],
        interaction_analysis: dict[str, Any],
        output_path: Path | None = None,
    ) -> Path:
        """Create comprehensive visualization of the analysis."""
        if output_path is None:
            output_path = self.data_dir / "composition_analysis.png"

        alphas = np.asarray(props_2d["alphas"], dtype=float)
        betas = np.asarray(props_2d["betas"], dtype=float)
        grid_actual = np.asarray(props_2d["loss_grid"], dtype=float)
        grid_add = np.asarray(interaction_analysis["loss_additive"], dtype=float)
        grid_inter = np.asarray(interaction_analysis["interaction"], dtype=float)

        # Robust scaling for actual/additive (shared)
        all_vals = np.concatenate([grid_actual.ravel(), grid_add.ravel()])
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size:
            vmin = float(np.percentile(all_vals, 1))
            vmax = float(np.percentile(all_vals, 99))
            if vmin >= vmax:
                vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        else:
            vmin, vmax = 0.0, 1.0

        # Symmetric limits for interaction around 0
        inter_finite = grid_inter[np.isfinite(grid_inter)]
        if inter_finite.size:
            vmax_inter = float(np.max(np.abs(inter_finite)))
            vmin_inter = -vmax_inter
        else:
            vmin_inter, vmax_inter = -1.0, 1.0

        # Mask NaNs for display
        A = np.ma.masked_invalid(grid_actual)  # noqa: N806 - Array variable
        A_add = np.ma.masked_invalid(grid_add)  # noqa: N806 - Array variable
        A_int = np.ma.masked_invalid(grid_inter)  # noqa: N806 - Array variable

        extent = (float(betas[0]), float(betas[-1]), float(alphas[0]), float(alphas[-1]))

        plt.figure(figsize=(16, 12))

        # 1. 2D heatmap with optimal point (actual)
        ax1 = plt.subplot(2, 3, 1)
        im1 = ax1.imshow(A, origin="lower", extent=extent, aspect="auto",
                         cmap="viridis", vmin=vmin, vmax=vmax)
        ax1.plot(props_2d["beta_opt"], props_2d["alpha_opt"], "r*", markersize=15,
                 label=f'Optimal ({props_2d["alpha_opt"]:.2f}, {props_2d["beta_opt"]:.2f})')
        ax1.axhline(0, color="white", linestyle="--", alpha=0.5, linewidth=1)
        ax1.axvline(0, color="white", linestyle="--", alpha=0.5, linewidth=1)
        ax1.set_xlabel("β (T2 scaling)")
        ax1.set_ylabel("α (T1 scaling)")
        ax1.set_title("2D Loss Landscape L(α, β)")
        ax1.legend()
        plt.colorbar(im1, ax=ax1, label="Loss")

        # 2. Cross-section at optimal β
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(cross_sections["alpha_range"], cross_sections["loss_at_beta_opt"],
                 "b-", label=f'L(α, β={props_2d["beta_opt"]:.2f})')
        ax2.plot(props_1d["alphas"], props_1d["losses"], "g--",
                 label="L(α, 0) [1D T1 only]", alpha=0.7)
        ax2.axvline(props_2d["alpha_opt"], color="red", linestyle="--",
                    label=f'2D optimal α={props_2d["alpha_opt"]:.2f}')
        ax2.axvline(props_1d["alpha_opt"], color="green", linestyle=":",
                    label=f'1D optimal α={props_1d["alpha_opt"]:.2f}')
        if np.isfinite(props_2d["base_loss"]):
            ax2.axhline(props_2d["base_loss"], color="gray", linestyle=":", alpha=0.5)
        ax2.set_xlabel("α")
        ax2.set_ylabel("Loss")
        ax2.set_title("Cross-section: Fix β at Optimal")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Cross-section at optimal α
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(cross_sections["beta_range"], cross_sections["loss_at_alpha_opt"],
                 "r-", label=f'L(α={props_2d["alpha_opt"]:.2f}, β)')
        ax3.axvline(props_2d["beta_opt"], color="red", linestyle="--",
                    label=f'2D optimal β={props_2d["beta_opt"]:.2f}')
        if np.isfinite(props_2d["base_loss"]):
            ax3.axhline(props_2d["base_loss"], color="gray", linestyle=":", alpha=0.5)
        ax3.set_xlabel("β")
        ax3.set_ylabel("Loss")
        ax3.set_title("Cross-section: Fix α at Optimal")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4. Interaction heatmap
        ax4 = plt.subplot(2, 3, 4)
        im4 = ax4.imshow(A_int, origin="lower", extent=extent, aspect="auto",
                         cmap="RdBu_r", vmin=vmin_inter, vmax=vmax_inter)
        ax4.plot(props_2d["beta_opt"], props_2d["alpha_opt"], "k*", markersize=15)
        ax4.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax4.axvline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax4.set_xlabel("β (T2 scaling)")
        ax4.set_ylabel("α (T1 scaling)")
        ax4.set_title(f'Interaction: L_actual - L_additive\nR²={interaction_analysis["r_squared"]:.3f}')
        plt.colorbar(im4, ax=ax4, label="Interaction")

        # 5. Additive model prediction (same scale as actual)
        ax5 = plt.subplot(2, 3, 5)
        im5 = ax5.imshow(A_add, origin="lower", extent=extent, aspect="auto",
                         cmap="viridis", vmin=vmin, vmax=vmax)
        ax5.plot(props_2d["beta_opt"], props_2d["alpha_opt"], "r*", markersize=15)
        ax5.axhline(0, color="white", linestyle="--", alpha=0.5, linewidth=1)
        ax5.axvline(0, color="white", linestyle="--", alpha=0.5, linewidth=1)
        ax5.set_xlabel("β (T2 scaling)")
        ax5.set_ylabel("α (T1 scaling)")
        ax5.set_title("Additive Model: L(α,0) + L(0,β) - L(0,0)")
        plt.colorbar(im5, ax=ax5, label="Loss")

        # 6. Comparison: axes cross-sections
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(alphas, cross_sections["loss_at_beta_0"], "g-", label="L(α, 0) [no T2]", linewidth=2)
        ax6.plot(betas, cross_sections["loss_at_alpha_0"], "b-", label="L(0, β) [no T1]", linewidth=2)
        ax6.axvline(props_2d["alpha_opt"], color="green", linestyle="--", alpha=0.5)
        ax6.axvline(props_2d["beta_opt"], color="blue", linestyle="--", alpha=0.5)
        if np.isfinite(props_2d["base_loss"]):
            ax6.axhline(props_2d["base_loss"], color="gray", linestyle=":", alpha=0.5)
        ax6.set_xlabel("α or β")
        ax6.set_ylabel("Loss")
        ax6.set_title("Individual Task Contributions")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path

    # --------------------------- pipeline ---------------------------

    def run_full_analysis(
        self,
        save_results: bool = True,
        save_visualization: bool = True,
    ) -> dict[str, Any]:
        """Run complete composition analysis pipeline."""
        print("=" * 80)
        print("ANALYSIS: Predicting Optimal Composition from Individual Task Properties")
        print("=" * 80)
        print()

        # Extract properties
        print("Extracting properties...")
        props_1d = self.extract_1d_properties()
        props_1d_t2 = self.extract_1d_properties(self.data_1d_t2) if self.data_1d_t2 is not None else None
        props_2d = self.extract_2d_properties()

        print("\n" + "=" * 80)
        print("1D TASK VECTOR (T1)")
        print("=" * 80)
        print(f"Optimal α: {props_1d['alpha_opt']:.4f}")
        print(f"Loss at optimal: {props_1d['loss_opt']:.4f}")
        print(f"Base loss (α≈0): {props_1d['base_loss']:.4f}")
        if props_1d["curvature"] is not None:
            print(f"Curvature at optimal: {props_1d['curvature']:.4f}")
        print(f"Zero-crossings: {[f'{z:.4f}' for z in props_1d['zero_crossings']]}")

        if props_1d_t2 is not None:
            print("\n" + "=" * 80)
            print("1D TASK VECTOR (T2)")
            print("=" * 80)
            print(f"Optimal α: {props_1d_t2['alpha_opt']:.4f}")
            print(f"Loss at optimal: {props_1d_t2['loss_opt']:.4f}")
            print(f"Base loss (α≈0): {props_1d_t2['base_loss']:.4f}")
            if props_1d_t2["curvature"] is not None:
                print(f"Curvature at optimal: {props_1d_t2['curvature']:.4f}")
            print(f"Zero-crossings: {[f'{z:.4f}' for z in props_1d_t2['zero_crossings']]}")

        print("\n" + "=" * 80)
        print("2D COMPOSITION")
        print("=" * 80)
        print(f"Optimal (α, β): ({props_2d['alpha_opt']:.4f}, {props_2d['beta_opt']:.4f})")
        print(f"Loss at optimal: {props_2d['loss_opt']:.4f}")
        if np.isfinite(props_2d["base_loss"]):
            print(f"Base loss L(0,0): {props_2d['base_loss']:.4f}")

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
        if np.isfinite(interaction_analysis["r_squared"]) and interaction_analysis["r_squared"] > 0.8:
            print("✓ Tasks are mostly independent (additive model fits well)")
        elif np.isfinite(interaction_analysis["r_squared"]) and interaction_analysis["r_squared"] > 0.5:
            print("⚠ Moderate interaction between tasks")
        else:
            print("✗ Strong interaction between tasks (non-additive)")

        # Make predictions
        print("\n" + "=" * 80)
        print("PREDICTIONS FROM 1D PROPERTIES")
        print("=" * 80)
        if props_1d_t2 is not None:
            print("Using T2 1D sweep data for improved 'independent' prediction")
        predictions = self.predict_optimal_from_1d(props_1d, props_1d_t2)
        actual = (props_2d["alpha_opt"], props_2d["beta_opt"])

        print(f"Actual 2D optimal: {actual}")
        print("\nPrediction strategies:")
        prediction_errors: dict[str, float] = {}
        for name, pred in predictions.items():
            error = float(np.sqrt((pred[0] - actual[0]) ** 2 + (pred[1] - actual[1]) ** 2))
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
                "curvature": float(props_1d["curvature"]) if props_1d["curvature"] is not None else None,
                "zero_crossings": [float(z) for z in props_1d["zero_crossings"]],
            },
            "1d_properties_t2": (
                {
                    "alpha_opt": float(props_1d_t2["alpha_opt"]),
                    "loss_opt": float(props_1d_t2["loss_opt"]),
                    "curvature": float(props_1d_t2["curvature"]) if props_1d_t2["curvature"] is not None else None,
                    "zero_crossings": [float(z) for z in props_1d_t2["zero_crossings"]],
                } if props_1d_t2 is not None else None
            ),
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
            "prediction_errors": {k: float(v) for k, v in prediction_errors.items()},
        }

        # Save results
        if save_results:
            output_path = self.data_dir / "composition_prediction_analysis.json"
            with open(output_path, "w") as f:
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
        "data_dir",
        type=str,
        help="Directory containing loss_landscape_results.json and loss_landscape_2d_results.json",
    )
    parser.add_argument(
        "--t2-1d",
        type=str,
        help="Path to T2 alpha sweep JSON (loss_landscape_results.json for T2). "
             "If not provided, will auto-detect 'loss_landscape_results_t2.json' in data_dir",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Skip saving JSON results",
    )
    parser.add_argument(
        "--no-save-viz",
        action="store_true",
        help="Skip saving visualization",
    )

    args = parser.parse_args()

    # Run analysis
    t2_1d_path = Path(args.t2_1d) if args.t2_1d else None
    analyzer = CompositionAnalyzer(Path(args.data_dir), t2_1d_path=t2_1d_path)
    analyzer.run_full_analysis(
        save_results=not args.no_save_results,
        save_visualization=not args.no_save_viz,
    )


if __name__ == "__main__":
    main()
