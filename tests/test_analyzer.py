"""Tests for result analyzer."""
import pytest
from sitv.data.models import AlphaSweepResult
from sitv.analysis.analyzer import ResultAnalyzer


class TestResultAnalyzer:
    """Tests for ResultAnalyzer class."""

    @pytest.fixture
    def sample_results(self):
        """Create sample alpha sweep results for testing."""
        results = [
            AlphaSweepResult(
                alpha=0.0,
                loss=4.5,
                base_loss=4.5,
                functional_return=0.0,
                task_performance=4.5,
            ),
            AlphaSweepResult(
                alpha=0.5,
                loss=3.2,
                base_loss=4.5,
                functional_return=1.3,
                task_performance=3.2,
            ),
            AlphaSweepResult(
                alpha=1.0,
                loss=2.8,  # Minimum loss
                base_loss=4.5,
                functional_return=1.7,
                task_performance=2.8,
            ),
            AlphaSweepResult(
                alpha=1.5,
                loss=3.5,
                base_loss=4.5,
                functional_return=1.0,
                task_performance=3.5,
            ),
            AlphaSweepResult(
                alpha=2.0,
                loss=4.45,  # Near zero-crossing
                base_loss=4.5,
                functional_return=0.05,
                task_performance=4.45,
            ),
        ]
        return results

    @pytest.fixture
    def results_with_squaring(self):
        """Create results with squaring test data."""
        results = [
            AlphaSweepResult(
                alpha=0.0,
                loss=4.5,
                base_loss=4.5,
                functional_return=0.0,
                task_performance=4.5,
                loss_2alpha=4.5,
                functional_return_2alpha=0.0,
            ),
            AlphaSweepResult(
                alpha=0.5,
                loss=3.2,
                base_loss=4.5,
                functional_return=1.3,
                task_performance=3.2,
                loss_2alpha=4.48,  # Squaring return point
                functional_return_2alpha=0.02,
            ),
            AlphaSweepResult(
                alpha=1.0,
                loss=2.8,
                base_loss=4.5,
                functional_return=1.7,
                task_performance=2.8,
                loss_2alpha=5.0,
                functional_return_2alpha=0.5,
            ),
        ]
        return results

    def test_initialization(self):
        """Should initialize with threshold."""
        analyzer = ResultAnalyzer(threshold=0.15)
        assert analyzer.threshold == 0.15

    def test_initialization_default_threshold(self):
        """Should use default threshold if not provided."""
        analyzer = ResultAnalyzer()
        assert analyzer.threshold == 0.1

    def test_analyze_finds_min_general_loss(self, sample_results):
        """Should find result with minimum general loss."""
        analyzer = ResultAnalyzer()
        analysis = analyzer.analyze(sample_results)

        assert "min_general_loss" in analysis
        assert analysis["min_general_loss"].alpha == 1.0
        assert analysis["min_general_loss"].loss == 2.8

    def test_analyze_finds_min_task_loss(self, sample_results):
        """Should find result with minimum task loss."""
        analyzer = ResultAnalyzer()
        analysis = analyzer.analyze(sample_results)

        assert "min_task_loss" in analysis
        assert analysis["min_task_loss"].alpha == 1.0
        assert analysis["min_task_loss"].task_performance == 2.8

    def test_analyze_finds_best_return(self, sample_results):
        """Should find result with best (smallest) functional return."""
        analyzer = ResultAnalyzer()
        analysis = analyzer.analyze(sample_results)

        assert "best_return" in analysis
        # Alpha=0 has functional_return=0, but we're looking for best non-zero
        # Actually best_return should be alpha=0 with return=0.0
        assert analysis["best_return"].alpha == 0.0
        assert analysis["best_return"].functional_return == 0.0

    def test_analyze_includes_sorted_by_return(self, sample_results):
        """Should include top 10 results sorted by functional return."""
        analyzer = ResultAnalyzer()
        analysis = analyzer.analyze(sample_results)

        assert "sorted_by_return" in analysis
        sorted_results = analysis["sorted_by_return"]

        # Should be sorted in ascending order of functional return
        for i in range(len(sorted_results) - 1):
            assert sorted_results[i].functional_return <= sorted_results[i+1].functional_return

    def test_analyze_includes_all_results(self, sample_results):
        """Should include all input results in output."""
        analyzer = ResultAnalyzer()
        analysis = analyzer.analyze(sample_results)

        assert "all_results" in analysis
        assert len(analysis["all_results"]) == len(sample_results)

    def test_find_zero_crossings(self, sample_results):
        """Should find zero-crossings where loss ≈ base_loss."""
        analyzer = ResultAnalyzer(threshold=0.1)
        analysis = analyzer.analyze(sample_results)

        zero_crossings = analysis["zero_crossings"]

        # Alpha=2.0 has functional_return=0.05 < threshold=0.1
        assert len(zero_crossings) == 1
        assert zero_crossings[0].alpha == 2.0

    def test_find_zero_crossings_with_custom_threshold(self, sample_results):
        """Should use custom threshold for zero-crossing detection."""
        # With stricter threshold, should find fewer crossings
        analyzer = ResultAnalyzer(threshold=0.01)
        analysis = analyzer.analyze(sample_results)

        zero_crossings = analysis["zero_crossings"]

        # Only alpha=0.0 has return < 0.01, but it's excluded (alpha must be > 0.15)
        assert len(zero_crossings) == 0

    def test_zero_crossings_excludes_near_zero_alpha(self):
        """Should exclude results with alpha ≈ 0 from zero-crossings."""
        results = [
            AlphaSweepResult(
                alpha=0.05,  # Too close to zero
                loss=4.48,
                base_loss=4.5,
                functional_return=0.02,
                task_performance=4.48,
            ),
            AlphaSweepResult(
                alpha=0.2,  # Far enough from zero
                loss=4.48,
                base_loss=4.5,
                functional_return=0.02,
                task_performance=4.48,
            ),
        ]

        analyzer = ResultAnalyzer(threshold=0.1)
        analysis = analyzer.analyze(results)

        # Should only include alpha=0.2
        assert len(analysis["zero_crossings"]) == 1
        assert analysis["zero_crossings"][0].alpha == 0.2

    def test_find_squaring_return_points(self, results_with_squaring):
        """Should find squaring return points where L(2α) ≈ L_base."""
        analyzer = ResultAnalyzer(threshold=0.1)
        analysis = analyzer.analyze(results_with_squaring)

        assert analysis["has_squaring_data"] is True
        assert len(analysis["squaring_return_points"]) == 1
        assert analysis["squaring_return_points"][0].alpha == 0.5

    def test_no_squaring_data(self, sample_results):
        """Should handle case with no squaring test data."""
        analyzer = ResultAnalyzer()
        analysis = analyzer.analyze(sample_results)

        assert analysis["has_squaring_data"] is False
        assert len(analysis["squaring_return_points"]) == 0

    def test_squaring_excludes_near_zero_alpha(self):
        """Should exclude results with alpha ≈ 0 from squaring return points."""
        results = [
            AlphaSweepResult(
                alpha=0.1,  # Too close to zero
                loss=3.0,
                base_loss=4.5,
                functional_return=1.5,
                task_performance=3.0,
                loss_2alpha=4.48,
                functional_return_2alpha=0.02,
            ),
            AlphaSweepResult(
                alpha=0.5,  # Far enough from zero
                loss=3.0,
                base_loss=4.5,
                functional_return=1.5,
                task_performance=3.0,
                loss_2alpha=4.48,
                functional_return_2alpha=0.02,
            ),
        ]

        analyzer = ResultAnalyzer(threshold=0.1)
        analysis = analyzer.analyze(results)

        # Should only include alpha=0.5
        assert len(analysis["squaring_return_points"]) == 1
        assert analysis["squaring_return_points"][0].alpha == 0.5

    def test_empty_results_list(self):
        """Should handle empty results list gracefully."""
        analyzer = ResultAnalyzer()

        # This might raise an exception, which is acceptable
        with pytest.raises(IndexError):
            analyzer.analyze([])

    def test_single_result(self):
        """Should handle single result."""
        results = [
            AlphaSweepResult(
                alpha=1.0,
                loss=3.0,
                base_loss=4.5,
                functional_return=1.5,
                task_performance=3.0,
            )
        ]

        analyzer = ResultAnalyzer()
        analysis = analyzer.analyze(results)

        assert analysis["min_general_loss"].alpha == 1.0
        assert analysis["min_task_loss"].alpha == 1.0
        assert len(analysis["all_results"]) == 1


class TestResultAnalyzerEdgeCases:
    """Tests for edge cases in result analysis."""

    def test_all_same_loss(self):
        """Should handle case where all losses are identical."""
        results = [
            AlphaSweepResult(
                alpha=float(i),
                loss=3.0,
                base_loss=3.0,
                functional_return=0.0,
                task_performance=3.0,
            )
            for i in range(5)
        ]

        analyzer = ResultAnalyzer()
        analysis = analyzer.analyze(results)

        # Should still find a minimum (first one)
        assert analysis["min_general_loss"] is not None
        assert analysis["min_general_loss"].loss == 3.0

    def test_monotonically_increasing_loss(self):
        """Should handle monotonically increasing loss."""
        results = [
            AlphaSweepResult(
                alpha=float(i),
                loss=3.0 + i * 0.5,
                base_loss=3.0,
                functional_return=i * 0.5,
                task_performance=3.0 + i * 0.5,
            )
            for i in range(5)
        ]

        analyzer = ResultAnalyzer()
        analysis = analyzer.analyze(results)

        # Minimum should be at alpha=0
        assert analysis["min_general_loss"].alpha == 0.0
        assert analysis["min_general_loss"].loss == 3.0

    def test_negative_alpha_values(self):
        """Should handle negative alpha values correctly."""
        results = [
            AlphaSweepResult(
                alpha=-1.0,
                loss=4.0,
                base_loss=4.5,
                functional_return=0.5,
                task_performance=4.0,
            ),
            AlphaSweepResult(
                alpha=0.0,
                loss=4.5,
                base_loss=4.5,
                functional_return=0.0,
                task_performance=4.5,
            ),
            AlphaSweepResult(
                alpha=1.0,
                loss=3.5,
                base_loss=4.5,
                functional_return=1.0,
                task_performance=3.5,
            ),
        ]

        analyzer = ResultAnalyzer()
        analysis = analyzer.analyze(results)

        # Should find minimum at alpha=1.0
        assert analysis["min_general_loss"].alpha == 1.0
        assert analysis["min_general_loss"].loss == 3.5
