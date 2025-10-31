"""Tests for markdown report generation."""
import pytest
from datetime import datetime
from sitv.data.models import AlphaSweepResult, ExperimentMetrics
from sitv.reporting.markdown import MarkdownReportGenerator


class TestMarkdownReportGenerator:
    """Tests for MarkdownReportGenerator."""

    @pytest.fixture
    def sample_results(self):
        """Create sample alpha sweep results."""
        return [
            AlphaSweepResult(
                alpha=0.0,
                loss=4.6544,
                base_loss=4.6544,
                functional_return=0.0,
                task_eval_loss=4.6544,
                loss_2alpha=4.6544,
                functional_return_2alpha=0.0,
            ),
            AlphaSweepResult(
                alpha=0.3826,
                loss=3.5,
                base_loss=4.6544,
                functional_return=1.1544,
                task_eval_loss=3.5,
                loss_2alpha=4.6638,
                functional_return_2alpha=0.009,  # Squaring return point
            ),
            AlphaSweepResult(
                alpha=1.0,
                loss=3.1425,
                base_loss=4.6544,
                functional_return=1.5119,
                task_eval_loss=3.1425,
                loss_2alpha=5.0,
                functional_return_2alpha=0.3456,
            ),
        ]

    @pytest.fixture
    def sample_metrics(self):
        """Create sample experiment metrics with all fields populated."""
        return ExperimentMetrics(
            start_time=datetime(2025, 1, 1, 10, 0, 0).isoformat(),
            end_time=datetime(2025, 1, 1, 10, 12, 24).isoformat(),
            duration_seconds=744.0,  # 12.4 minutes
            model_name="gpt2",
            device="cuda",
            model_parameters=124000000,
            finetuning_start_time=datetime(2025, 1, 1, 10, 0, 0).isoformat(),
            finetuning_end_time=datetime(2025, 1, 1, 10, 1, 12).isoformat(),
            finetuning_duration_seconds=72.0,  # 1.2 minutes
            training_examples=3000,
            num_epochs=2,
            learning_rate=5e-5,
            final_training_loss=0.2384,
            training_steps=376,
            training_history=[
                {"step": 1, "epoch": 0.0, "loss": 1.5, "learning_rate": 5e-5, "grad_norm": 2.1},
                {"step": 50, "epoch": 0.5, "loss": 0.8, "learning_rate": 5e-5, "grad_norm": 1.5},
                {"step": 188, "epoch": 1.0, "loss": 0.4, "learning_rate": 5e-5, "grad_norm": 1.2},
                {"step": 376, "epoch": 2.0, "loss": 0.2384, "learning_rate": 5e-5, "grad_norm": 0.9},
            ],
            task_vector_magnitude=14.24,
            task_vector_computation_time=0.5,
            sweep_start_time=datetime(2025, 1, 1, 10, 1, 12).isoformat(),
            sweep_end_time=datetime(2025, 1, 1, 10, 11, 24).isoformat(),
            sweep_duration_seconds=612.0,  # 10.2 minutes
            num_alpha_samples=150,
            alpha_range=(-3.0, 3.0),
            time_per_alpha_seconds=4.08,
            min_general_loss_alpha=0.2617,
            min_general_loss=3.1425,
            min_task_loss_alpha=0.2617,
            min_task_loss=3.1425,
            num_zero_crossings=0,
            zero_crossing_alphas=[],
            enable_squaring_test=True,
            num_squaring_return_points=1,
            squaring_return_alphas=[0.3826],
            task_name="sentiment",
        )

    @pytest.fixture
    def sample_analysis(self):
        """Create sample analysis dictionary."""
        return {
            "min_general_loss": AlphaSweepResult(
                alpha=0.2617,
                loss=3.1425,
                base_loss=4.6544,
                functional_return=1.5119,
                task_eval_loss=3.1425,
                loss_2alpha=5.0,
                functional_return_2alpha=0.3456,
            ),
            "min_task_loss": AlphaSweepResult(
                alpha=0.2617,
                loss=3.1425,
                base_loss=4.6544,
                functional_return=1.5119,
                task_eval_loss=3.1425,
                loss_2alpha=5.0,
                functional_return_2alpha=0.3456,
            ),
            "zero_crossings": [],
            "has_squaring_data": True,
            "squaring_return_points": [
                AlphaSweepResult(
                    alpha=0.3826,
                    loss=3.5,
                    base_loss=4.6544,
                    functional_return=1.1544,
                    task_eval_loss=3.5,
                    loss_2alpha=4.6638,
                    functional_return_2alpha=0.009,
                )
            ],
        }

    def test_duration_is_not_zero(self, sample_results, sample_analysis, sample_metrics, tmp_path):
        """Test that total duration is properly calculated and not 0.0."""
        generator = MarkdownReportGenerator()
        report_path = str(tmp_path / "test_report.md")

        generator.generate(sample_results, sample_analysis, sample_metrics, report_path)

        with open(report_path, 'r') as f:
            content = f.read()

        # Check that duration is 12.4 minutes, not 0.0
        assert "**Total Duration**: 12.4 minutes" in content
        assert "**Total Duration**: 0.0 minutes" not in content

    def test_timing_breakdown_percentages(self, sample_results, sample_analysis, sample_metrics, tmp_path):
        """Test that timing breakdown percentages are calculated correctly."""
        generator = MarkdownReportGenerator()
        report_path = str(tmp_path / "test_report.md")

        generator.generate(sample_results, sample_analysis, sample_metrics, report_path)

        with open(report_path, 'r') as f:
            content = f.read()

        # Check that percentages are not 0.0%
        assert "| Fine-tuning | 1.2m | 9.7%" in content  # 72/744 * 100 = 9.7%
        assert "| Alpha Sweep | 10.2m | 82.3%" in content  # 612/744 * 100 = 82.3%
        assert "| **Total** | **12.4m** | **100%** |" in content

    def test_squaring_return_points_in_conclusion(self, sample_results, sample_analysis, sample_metrics, tmp_path):
        """Test that squaring return points are considered in the conclusion."""
        generator = MarkdownReportGenerator()
        report_path = str(tmp_path / "test_report.md")

        generator.generate(sample_results, sample_analysis, sample_metrics, report_path)

        with open(report_path, 'r') as f:
            content = f.read()

        # Should mention squaring return points, not say "No - loss is monotonic"
        assert "No - loss is monotonic, no rotation-like symmetry detected" not in content
        assert "squaring return points" in content.lower()
        assert "rotation-like symmetry" in content

    def test_training_history_included(self, sample_results, sample_analysis, sample_metrics, tmp_path):
        """Test that training history is included in the report."""
        generator = MarkdownReportGenerator()
        report_path = str(tmp_path / "test_report.md")

        generator.generate(sample_results, sample_analysis, sample_metrics, report_path)

        with open(report_path, 'r') as f:
            content = f.read()

        # Should not say "No training history available"
        assert "**No training history available.**" not in content

        # Should include training history section with data
        assert "## Training History" in content
        assert "Training Progress" in content
        assert "| Step | Epoch | Loss | Learning Rate | Grad Norm |" in content

        # Should include some training data
        assert "| 1 |" in content  # First step
        assert "| 376 |" in content  # Last step
        assert "0.2384" in content  # Final loss

    def test_all_issues_fixed_together(self, sample_results, sample_analysis, sample_metrics, tmp_path):
        """Integration test that all issues are fixed together."""
        generator = MarkdownReportGenerator()
        report_path = str(tmp_path / "test_report.md")

        generator.generate(sample_results, sample_analysis, sample_metrics, report_path)

        with open(report_path, 'r') as f:
            content = f.read()

        # Issue 1: Duration should not be 0.0
        assert "**Total Duration**: 12.4 minutes" in content

        # Issue 2: Squaring return points should be acknowledged
        assert "Partial - found squaring return points" in content or "Yes - found" in content

        # Issue 3: Training history should be present
        assert "## Training History" in content
        assert "| 376 |" in content

        # Issue 4: Timing percentages should be calculated
        assert "9.7%" in content  # Fine-tuning percentage
        assert "82.3%" in content  # Alpha sweep percentage

    def test_empty_training_history_fallback(self, sample_results, sample_analysis, tmp_path):
        """Test that empty training history shows appropriate message."""
        # Create metrics with no training history
        metrics = ExperimentMetrics(
            start_time=datetime(2025, 1, 1, 10, 0, 0).isoformat(),
            duration_seconds=744.0,
            model_name="gpt2",
            training_history=[],  # Empty!
        )

        generator = MarkdownReportGenerator()
        report_path = str(tmp_path / "test_report.md")

        generator.generate(sample_results, sample_analysis, metrics, report_path)

        with open(report_path, 'r') as f:
            content = f.read()

        # Should show the fallback message
        assert "**No training history available.**" in content

    def test_zero_duration_fallback(self, sample_results, sample_analysis, tmp_path):
        """Test that zero duration is handled gracefully."""
        # Create metrics with zero duration
        metrics = ExperimentMetrics(
            start_time=datetime(2025, 1, 1, 10, 0, 0).isoformat(),
            duration_seconds=0.0,  # Zero!
            finetuning_duration_seconds=0.0,
            sweep_duration_seconds=0.0,
            model_name="gpt2",
        )

        generator = MarkdownReportGenerator()
        report_path = str(tmp_path / "test_report.md")

        generator.generate(sample_results, sample_analysis, metrics, report_path)

        with open(report_path, 'r') as f:
            content = f.read()

        # Should show 0.0 without crashing
        assert "**Total Duration**: 0.0 minutes" in content
        # Percentages should be 0.0% to avoid division by zero
        assert "0.0%" in content
