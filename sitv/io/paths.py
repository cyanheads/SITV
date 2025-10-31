"""
Path management utilities for SITV.

This module provides utilities for managing file paths and directories.
"""

import os
from typing import Optional


class PathManager:
    """Service for managing file paths and directories.

    This manager provides utilities for:
    - Constructing output file paths
    - Creating directories
    - Managing output directory structure

    Attributes:
        output_dir: Base output directory for all files
    """

    def __init__(self, output_dir: str = "outputs"):
        """Initialize the path manager.

        Args:
            output_dir: Base output directory
        """
        self.output_dir = output_dir

    def ensure_output_dir(self) -> str:
        """Ensure output directory exists.

        Returns:
            Path to output directory

        Examples:
            >>> pm = PathManager()
            >>> output_dir = pm.ensure_output_dir()
        """
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    def get_plot_path(self, filename: str = "loss_landscape_sweep.png") -> str:
        """Get path for plot output.

        Args:
            filename: Plot filename

        Returns:
            Full path to plot file

        Examples:
            >>> pm = PathManager("outputs")
            >>> path = pm.get_plot_path("results.png")
        """
        return os.path.join(self.output_dir, filename)

    def get_report_path(self, filename: str = "experiment_report.md") -> str:
        """Get path for markdown report.

        Args:
            filename: Report filename

        Returns:
            Full path to report file

        Examples:
            >>> pm = PathManager("outputs")
            >>> path = pm.get_report_path()
        """
        return os.path.join(self.output_dir, filename)

    def get_results_path(self, filename: str = "loss_landscape_results.json") -> str:
        """Get path for JSON results.

        Args:
            filename: Results filename

        Returns:
            Full path to results file

        Examples:
            >>> pm = PathManager("outputs")
            >>> path = pm.get_results_path()
        """
        return os.path.join(self.output_dir, filename)

    def get_metrics_path(self, filename: str = "experiment_metrics.json") -> str:
        """Get path for experiment metrics.

        Args:
            filename: Metrics filename

        Returns:
            Full path to metrics file

        Examples:
            >>> pm = PathManager("outputs")
            >>> path = pm.get_metrics_path()
        """
        return os.path.join(self.output_dir, filename)

    def get_model_save_path(self, model_type: str) -> str:
        """Get path for saved model.

        Args:
            model_type: Type of model ("base" or "finetuned")

        Returns:
            Full path to model directory

        Examples:
            >>> pm = PathManager("outputs")
            >>> path = pm.get_model_save_path("base")
        """
        if model_type == "base":
            return os.path.join(self.output_dir, "saved_base_model")
        elif model_type == "finetuned":
            return os.path.join(self.output_dir, "saved_finetuned_model")
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

    def list_output_files(self) -> list[str]:
        """List all files in output directory.

        Returns:
            List of filenames in output directory

        Examples:
            >>> pm = PathManager("outputs")
            >>> files = pm.list_output_files()
        """
        if not os.path.exists(self.output_dir):
            return []
        return os.listdir(self.output_dir)
