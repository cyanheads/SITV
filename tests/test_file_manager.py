"""Tests for file manager."""
import pytest
import json
import os
from pathlib import Path
from sitv.data.models import AlphaSweepResult, TwoDSweepResult, ExperimentMetrics
from sitv.io.file_manager import FileManager


class TestFileManager:
    """Tests for FileManager class."""

    def test_initialization(self, tmp_path):
        """Should initialize and create output directory."""
        output_dir = str(tmp_path / "test_output")
        fm = FileManager(output_dir=output_dir)

        assert fm.output_dir == output_dir
        assert os.path.exists(output_dir)

    def test_initialization_creates_nested_directories(self, tmp_path):
        """Should create nested directories if they don't exist."""
        output_dir = str(tmp_path / "level1" / "level2" / "level3")
        fm = FileManager(output_dir=output_dir)

        assert os.path.exists(output_dir)

    def test_save_results(self, tmp_path):
        """Should save alpha sweep results to JSON file."""
        fm = FileManager(output_dir=str(tmp_path))

        results = [
            AlphaSweepResult(
                alpha=0.0,
                loss=4.5,
                base_loss=4.5,
                functional_return=0.0,
                task_performance=4.5,
            ),
            AlphaSweepResult(
                alpha=1.0,
                loss=3.2,
                base_loss=4.5,
                functional_return=1.3,
                task_performance=3.2,
            ),
        ]

        filepath = fm.save_results(results)

        # Check file was created
        assert os.path.exists(filepath)

        # Check file contains correct data
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)

        assert len(loaded_data) == 2
        assert loaded_data[0]["alpha"] == 0.0
        assert loaded_data[1]["alpha"] == 1.0
        assert loaded_data[1]["loss"] == 3.2

    def test_save_results_custom_filename(self, tmp_path):
        """Should save with custom filename."""
        fm = FileManager(output_dir=str(tmp_path))

        results = [
            AlphaSweepResult(
                alpha=0.0,
                loss=4.5,
                base_loss=4.5,
                functional_return=0.0,
                task_performance=4.5,
            )
        ]

        filepath = fm.save_results(results, filename="custom_results.json")

        assert filepath.endswith("custom_results.json")
        assert os.path.exists(filepath)

    def test_save_2d_results(self, tmp_path):
        """Should save 2D composition results to JSON file."""
        fm = FileManager(output_dir=str(tmp_path))

        results = [
            TwoDSweepResult(
                alpha=0.0,
                beta=0.0,
                loss=4.5,
                base_loss=4.5,
                functional_return=0.0,
                perplexity=90.0,
            ),
            TwoDSweepResult(
                alpha=1.0,
                beta=0.5,
                loss=3.2,
                base_loss=4.5,
                functional_return=1.3,
                perplexity=24.5,
            ),
        ]

        filepath = fm.save_2d_results(results)

        # Check file was created
        assert os.path.exists(filepath)

        # Check file contains correct data
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)

        assert len(loaded_data) == 2
        assert loaded_data[0]["alpha"] == 0.0
        assert loaded_data[0]["beta"] == 0.0
        assert loaded_data[1]["alpha"] == 1.0
        assert loaded_data[1]["beta"] == 0.5

    def test_save_metrics(self, tmp_path):
        """Should save experiment metrics to JSON file."""
        fm = FileManager(output_dir=str(tmp_path))

        metrics = ExperimentMetrics(
            start_time="2025-01-01T00:00:00",
            end_time="2025-01-01T01:00:00",
            duration_seconds=3600.0,
            model_name="test-model",
            num_alpha_samples=100,
        )

        filepath = fm.save_metrics(metrics)

        # Check file was created
        assert os.path.exists(filepath)

        # Check file contains correct data
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data["model_name"] == "test-model"
        assert loaded_data["duration_seconds"] == 3600.0
        assert loaded_data["num_alpha_samples"] == 100

    def test_file_exists(self, tmp_path):
        """Should check if file exists in output directory."""
        fm = FileManager(output_dir=str(tmp_path))

        # Create a test file
        test_file = tmp_path / "test.json"
        test_file.write_text("{}")

        assert fm.file_exists("test.json") is True
        assert fm.file_exists("nonexistent.json") is False

    def test_get_path(self, tmp_path):
        """Should return full path for filename."""
        fm = FileManager(output_dir=str(tmp_path))

        path = fm.get_path("test.json")

        assert path == str(tmp_path / "test.json")

    def test_save_json_general(self, tmp_path):
        """Should save arbitrary JSON data."""
        fm = FileManager(output_dir=str(tmp_path))

        data = {
            "key1": "value1",
            "key2": 42,
            "key3": [1, 2, 3],
        }

        filepath = fm.save_json(data, "test.json")

        # Check file was created
        assert os.path.exists(filepath)

        # Check file contains correct data
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data == data

    def test_load_json(self, tmp_path):
        """Should load JSON data from file."""
        fm = FileManager(output_dir=str(tmp_path))

        # Create a test JSON file
        data = {"test": "data", "number": 123}
        test_file = tmp_path / "test.json"

        with open(test_file, 'w') as f:
            json.dump(data, f)

        # Load the data
        loaded_data = fm.load_json("test.json")

        assert loaded_data == data

    def test_load_json_nonexistent_file(self, tmp_path):
        """Should raise error for nonexistent file."""
        fm = FileManager(output_dir=str(tmp_path))

        with pytest.raises(FileNotFoundError):
            fm.load_json("nonexistent.json")

    def test_json_indent_formatting(self, tmp_path):
        """Should save JSON with proper indentation."""
        fm = FileManager(output_dir=str(tmp_path))

        results = [
            AlphaSweepResult(
                alpha=0.0,
                loss=4.5,
                base_loss=4.5,
                functional_return=0.0,
                task_performance=4.5,
            )
        ]

        filepath = fm.save_results(results)

        # Read file content
        with open(filepath, 'r') as f:
            content = f.read()

        # Check that it's indented (not single line)
        assert '\n' in content
        assert '  ' in content  # Should have 2-space indentation

    def test_handles_empty_results_list(self, tmp_path):
        """Should handle saving empty results list."""
        fm = FileManager(output_dir=str(tmp_path))

        filepath = fm.save_results([])

        # Check file was created
        assert os.path.exists(filepath)

        # Check file contains empty array
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data == []


class TestFileManagerIntegration:
    """Integration tests for FileManager."""

    def test_save_and_load_round_trip(self, tmp_path):
        """Should successfully save and load data."""
        fm = FileManager(output_dir=str(tmp_path))

        original_data = {
            "experiment": "test",
            "results": [1, 2, 3],
            "metrics": {"loss": 0.5}
        }

        # Save
        fm.save_json(original_data, "test.json")

        # Load
        loaded_data = fm.load_json("test.json")

        # Compare
        assert loaded_data == original_data

    def test_multiple_files_in_same_directory(self, tmp_path):
        """Should handle multiple files in same directory."""
        fm = FileManager(output_dir=str(tmp_path))

        # Save multiple files
        fm.save_json({"file": 1}, "file1.json")
        fm.save_json({"file": 2}, "file2.json")
        fm.save_json({"file": 3}, "file3.json")

        # Check all exist
        assert fm.file_exists("file1.json")
        assert fm.file_exists("file2.json")
        assert fm.file_exists("file3.json")

        # Load and verify
        assert fm.load_json("file1.json")["file"] == 1
        assert fm.load_json("file2.json")["file"] == 2
        assert fm.load_json("file3.json")["file"] == 3
