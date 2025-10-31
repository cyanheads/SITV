"""
Result analysis service for SITV experiments.

This module provides the ResultAnalyzer for analyzing experiment results,
finding optimal alpha values, zero-crossings, and special properties.
"""

from typing import Any, Dict, List

from sitv.data.models import AlphaSweepResult


class ResultAnalyzer:
    """Service for analyzing experiment results.

    This analyzer processes alpha sweep results to find:
    - Minimum general loss and corresponding α
    - Minimum task-specific loss and corresponding α
    - Zero-crossings: where L(α) ≈ L(M_base) for α ≠ 0
    - Squaring return points: where L(2α) ≈ L(M_base)
    - Special α values with small functional return

    Attributes:
        threshold: Functional return threshold for zero-crossing detection
    """

    def __init__(self, threshold: float = 0.1):
        """Initialize the result analyzer.

        Args:
            threshold: Functional return threshold for "crossing" detection
        """
        self.threshold = threshold

    def analyze(self, results: List[AlphaSweepResult]) -> Dict[str, Any]:
        """Analyze loss landscape sweep results.

        Args:
            results: List of AlphaSweepResult objects from experiment

        Returns:
            Dictionary containing analysis results:
                - min_general_loss: Result with minimum general loss
                - min_task_loss: Result with minimum task loss
                - best_return: Result with smallest functional return
                - zero_crossings: List of zero-crossing points
                - sorted_by_return: Top 10 results sorted by functional return
                - squaring_return_points: Squaring test return points (if available)
                - has_squaring_data: Whether squaring test data exists
                - all_results: All input results

        Examples:
            >>> analyzer = ResultAnalyzer(threshold=0.1)
            >>> analysis = analyzer.analyze(sweep_results)
            >>> print(f"Best alpha: {analysis['min_general_loss'].alpha}")
        """
        # Sort results by different criteria
        sorted_by_loss = sorted(results, key=lambda r: r.loss)
        sorted_by_task_loss = sorted(results, key=lambda r: r.task_eval_loss)
        sorted_by_return = sorted(results, key=lambda r: r.functional_return)

        # Find zero-crossings
        zero_crossings = self._find_zero_crossings(results)

        # Find squaring return points
        squaring_return_points, has_squaring_data = self._find_squaring_return_points(results)

        # Print analysis
        self._print_analysis(
            sorted_by_loss,
            sorted_by_task_loss,
            sorted_by_return,
            zero_crossings,
            squaring_return_points,
            has_squaring_data
        )

        return {
            "min_general_loss": sorted_by_loss[0],
            "min_task_loss": sorted_by_task_loss[0],
            "best_return": sorted_by_return[0],
            "zero_crossings": zero_crossings,
            "sorted_by_return": sorted_by_return[:10],
            "squaring_return_points": squaring_return_points,
            "has_squaring_data": has_squaring_data,
            "all_results": results,
        }

    def _find_zero_crossings(self, results: List[AlphaSweepResult]) -> List[AlphaSweepResult]:
        """Find zero-crossings where L(α) ≈ L_base for α ≠ 0.

        Args:
            results: List of results to analyze

        Returns:
            List of results at zero-crossings
        """
        zero_crossings = []
        for result in results:
            if abs(result.alpha) > 0.15 and result.functional_return < self.threshold:
                zero_crossings.append(result)
        return zero_crossings

    def _find_squaring_return_points(
        self,
        results: List[AlphaSweepResult]
    ) -> tuple[List[AlphaSweepResult], bool]:
        """Find squaring return points where L(2α) ≈ L_base.

        Args:
            results: List of results to analyze

        Returns:
            Tuple of (squaring_return_points, has_squaring_data)
        """
        # Check if squaring test data exists
        has_squaring_data = all(
            r.loss_2alpha != 0.0 for r in results if r.alpha != 0
        )

        if not has_squaring_data:
            return [], False

        squaring_return_points = []
        for result in results:
            if abs(result.alpha) > 0.15 and result.functional_return_2alpha < self.threshold:
                squaring_return_points.append(result)

        return squaring_return_points, True

    def _print_analysis(
        self,
        sorted_by_loss: List[AlphaSweepResult],
        sorted_by_task_loss: List[AlphaSweepResult],
        sorted_by_return: List[AlphaSweepResult],
        zero_crossings: List[AlphaSweepResult],
        squaring_return_points: List[AlphaSweepResult],
        has_squaring_data: bool
    ) -> None:
        """Print analysis results to console.

        Args:
            sorted_by_loss: Results sorted by general loss
            sorted_by_task_loss: Results sorted by task loss
            sorted_by_return: Results sorted by functional return
            zero_crossings: Zero-crossing results
            squaring_return_points: Squaring return point results
            has_squaring_data: Whether squaring data is available
        """
        print(f"\n{'='*70}")
        print("LOSS LANDSCAPE ANALYSIS")
        print(f"{'='*70}\n")

        # Minimum general loss
        min_general_result = sorted_by_loss[0]
        print("Minimum General Loss (best general knowledge):")
        print(f"  α = {min_general_result.alpha:+.4f}")
        print(f"  L(α) = {min_general_result.loss:.4f}")
        print(f"  L(M_base) = {min_general_result.base_loss:.4f}")
        print(f"  Δ = {min_general_result.loss - min_general_result.base_loss:+.4f}\n")

        # Minimum task-specific evaluation loss
        min_task_result = sorted_by_task_loss[0]
        print("Minimum Task-Specific Loss (best on task evaluation data):")
        print(f"  α = {min_task_result.alpha:+.4f}")
        print(f"  Task Eval L(α) = {min_task_result.task_eval_loss:.4f}")
        print(f"  General L(α) = {min_task_result.loss:.4f}")
        print(
            f"  Δ from base = {min_task_result.task_eval_loss - min_task_result.base_loss:+.4f}\n"
        )

        # Best functional return
        print("Best Functional Return (smallest |L(α) - L_base|):")
        for i, result in enumerate(sorted_by_return[:5], 1):
            print(f"  {i}. α = {result.alpha:+.4f}, |ΔL| = {result.functional_return:.6f}")

        # Zero-crossings
        print("\nZero-Crossings (where L(α) ≈ L_base for α ≠ 0):")
        if zero_crossings:
            print(f"  Found {len(zero_crossings)} crossing(s):")
            for i, result in enumerate(zero_crossings[:5], 1):
                print(
                    f"  {i}. α = {result.alpha:+.4f}, L(α) = {result.loss:.4f}, "
                    f"|ΔL| = {result.functional_return:.6f} ★"
                )
        else:
            print(f"  No zero-crossings found (threshold: |ΔL| < {self.threshold})")
            print("  → Loss is monotonic along task vector direction")

        # Squaring test analysis
        if has_squaring_data:
            self._print_squaring_analysis(squaring_return_points)

    def _print_squaring_analysis(
        self,
        squaring_return_points: List[AlphaSweepResult]
    ) -> None:
        """Print squaring test analysis.

        Args:
            squaring_return_points: List of squaring return point results
        """
        print("\n" + "="*70)
        print("SQUARING TEST ANALYSIS: [W(λ)]² = I Analog")
        print("="*70)
        print("\nSquaring Return Points (where L(2α) ≈ L_base for α ≠ 0):")

        if squaring_return_points:
            print(f" ★ Found {len(squaring_return_points)} squaring return point(s)!")
            print("  → These α values exhibit the self-inverse property: doubling brings back to base loss")
            for i, result in enumerate(squaring_return_points[:5], 1):
                print(
                    f"  {i}. α = {result.alpha:+.4f}, L(2α) = {result.loss_2alpha:.4f}, "
                    f"|ΔL(2α)| = {result.functional_return_2alpha:.6f} ★"
                )
            print("\n  INTERPRETATION:")
            print("  This suggests neural loss landscapes MAY exhibit rotation-like symmetry!")
            print("  Analogous to R(n,π)² = I in rotation groups.")
        else:
            print(f"  No squaring return points found (threshold: |ΔL(2α)| < {self.threshold})")
            print("  → Neural loss landscapes do not exhibit self-inverse property under doubling")
            print("  → Unlike rotation groups, no [W(λ)]² = I analog detected")
