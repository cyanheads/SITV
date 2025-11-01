"""
File management service for SITV.

This module provides the FileManager for handling all file I/O operations
including JSON serialization and file writing.
"""

import json
import os
from typing import Any, Dict, List
from dataclasses import asdict

from sitv.data.models import AlphaSweepResult, TwoDSweepResult, ExperimentMetrics


class FileManager:
    """Service for file I/O operations.

    This manager handles:
    - JSON serialization of results
    - Saving experiment data
    - Loading saved results
    - File existence checks

    Attributes:
        output_dir: Base output directory
    """

    def __init__(self, output_dir: str = "outputs"):
        """Initialize the file manager.

        Args:
            output_dir: Base output directory for files
        """
        self.output_dir = output_dir
        self._ensure_output_dir()

    def _ensure_output_dir(self) -> None:
        """Ensure output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)

    def save_results(
        self, results: List[AlphaSweepResult], filename: str = "loss_landscape_results.json"
    ) -> str:
        """Save alpha sweep results to JSON.

        Args:
            results: List of AlphaSweepResult objects
            filename: Output filename

        Returns:
            Path to saved file

        Examples:
            >>> fm = FileManager()
            >>> path = fm.save_results(results)
        """
        filepath = os.path.join(self.output_dir, filename)

        # Convert to dictionaries
        results_dict = [asdict(r) for r in results]

        # Save to JSON
        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved: {filepath}")
        return filepath

    def save_2d_results(
        self, results: List[TwoDSweepResult], filename: str = "loss_landscape_2d_results.json"
    ) -> str:
        """Save 2D composition results to JSON.

        Args:
            results: List of TwoDSweepResult objects
            filename: Output filename

        Returns:
            Path to saved file

        Examples:
            >>> fm = FileManager()
            >>> path = fm.save_2d_results(results_2d)
        """
        filepath = os.path.join(self.output_dir, filename)

        # Convert to dictionaries
        results_dict = [asdict(r) for r in results]

        # Save to JSON
        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"2D results saved: {filepath}")
        return filepath

    def save_metrics(
        self, metrics: ExperimentMetrics, filename: str = "experiment_metrics.json"
    ) -> str:
        """Save experiment metrics to JSON.

        Args:
            metrics: ExperimentMetrics object
            filename: Output filename

        Returns:
            Path to saved file

        Examples:
            >>> fm = FileManager()
            >>> path = fm.save_metrics(metrics)
        """
        filepath = os.path.join(self.output_dir, filename)

        # Convert to dictionary
        metrics_dict = asdict(metrics)

        # Save to JSON
        with open(filepath, "w") as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"Metrics saved: {filepath}")
        return filepath

    def save_analysis(
        self, analysis: Dict[str, Any], filename: str = "analysis_results.json"
    ) -> str:
        """Save analysis results to JSON.

        Args:
            analysis: Analysis dictionary
            filename: Output filename

        Returns:
            Path to saved file

        Examples:
            >>> fm = FileManager()
            >>> path = fm.save_analysis(analysis_dict)
        """
        filepath = os.path.join(self.output_dir, filename)

        # Convert AlphaSweepResult objects to dicts
        serializable_analysis: Dict[str, Any] = {}
        for key, value in analysis.items():
            if isinstance(value, AlphaSweepResult):
                serializable_analysis[key] = asdict(value)
            elif isinstance(value, list) and value and isinstance(value[0], AlphaSweepResult):
                serializable_analysis[key] = [asdict(r) for r in value]
            elif key == "all_results":
                # Skip to avoid duplication
                continue
            else:
                serializable_analysis[key] = value

        # Save to JSON
        with open(filepath, "w") as f:
            json.dump(serializable_analysis, f, indent=2)

        print(f"Analysis saved: {filepath}")
        return filepath

    def load_results(self, filename: str = "loss_landscape_results.json") -> List[Dict[str, Any]]:
        """Load results from JSON file.

        Args:
            filename: Input filename

        Returns:
            List of result dictionaries

        Examples:
            >>> fm = FileManager()
            >>> results = fm.load_results()
        """
        filepath = os.path.join(self.output_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results file not found: {filepath}")

        with open(filepath, "r") as f:
            results: List[Dict[str, Any]] = json.load(f)

        return results

    def file_exists(self, filename: str) -> bool:
        """Check if a file exists in output directory.

        Args:
            filename: Filename to check

        Returns:
            True if file exists, False otherwise

        Examples:
            >>> fm = FileManager()
            >>> if fm.file_exists("results.json"):
            ...     print("Results found")
        """
        filepath = os.path.join(self.output_dir, filename)
        return os.path.exists(filepath)

    def get_output_summary(self) -> Dict[str, Any]:
        """Get summary of output directory contents.

        Returns:
            Dictionary with file counts and sizes

        Examples:
            >>> fm = FileManager()
            >>> summary = fm.get_output_summary()
            >>> print(f"Total files: {summary['file_count']}")
        """
        if not os.path.exists(self.output_dir):
            return {"file_count": 0, "files": []}

        files = []
        total_size = 0

        for filename in os.listdir(self.output_dir):
            filepath = os.path.join(self.output_dir, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                files.append({"name": filename, "size": size})
                total_size += size

        return {
            "file_count": len(files),
            "total_size_bytes": total_size,
            "files": files,
        }
