"""
Dataset loader for SITV experiments.

This module provides utilities for loading training and evaluation datasets
from text files in the data/ directory.
"""

import os
from pathlib import Path
from typing import List


class DatasetLoader:
    """Utility class for loading datasets from text files.

    Supports three types of datasets:
    1. General evaluation datasets (data/general/)
    2. Task training datasets (data/tasks/)
    3. Task evaluation datasets (data/eval/)

    File format:
    - One example per line
    - Lines starting with # are comments
    - Empty lines are ignored
    - UTF-8 encoding
    """

    def __init__(self, data_dir: str = None):
        """Initialize the dataset loader.

        Args:
            data_dir: Root data directory path. If None, uses ./data relative
                     to project root.
        """
        if data_dir is None:
            # Default to data/ directory in project root
            project_root = Path(__file__).parent.parent.parent
            self.data_dir = project_root / "data"
        else:
            self.data_dir = Path(data_dir)

        self.general_dir = self.data_dir / "general"
        self.tasks_dir = self.data_dir / "tasks"
        self.eval_dir = self.data_dir / "eval"

    def _load_file(self, file_path: Path) -> List[str]:
        """Load examples from a text file.

        Args:
            file_path: Path to the text file

        Returns:
            List of text examples (one per line)

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {file_path}\n"
                f"Expected location: {file_path.absolute()}"
            )

        examples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Strip whitespace
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                examples.append(line)

        if not examples:
            raise ValueError(
                f"No examples found in {file_path}. "
                f"File may be empty or contain only comments."
            )

        return examples

    def load_general(self, dataset_name: str) -> List[str]:
        """Load a general evaluation dataset.

        Args:
            dataset_name: Name of the dataset (without .txt extension)
                         e.g., "mixed_domain", "wikitext", "coding"

        Returns:
            List of evaluation text examples

        Example:
            >>> loader = DatasetLoader()
            >>> texts = loader.load_general("mixed_domain")
        """
        file_path = self.general_dir / f"{dataset_name}.txt"
        return self._load_file(file_path)

    def load_task(self, task_name: str) -> List[str]:
        """Load task training data.

        Args:
            task_name: Name of the task (without .txt extension)
                      e.g., "sentiment_positive", "qa_factual"

        Returns:
            List of training text examples

        Example:
            >>> loader = DatasetLoader()
            >>> texts = loader.load_task("sentiment_positive")
        """
        file_path = self.tasks_dir / f"{task_name}.txt"
        return self._load_file(file_path)

    def load_eval(self, task_name: str) -> List[str]:
        """Load task evaluation data.

        Args:
            task_name: Name of the task evaluation file (without .txt)
                      e.g., "sentiment_positive_eval"

        Returns:
            List of evaluation text examples

        Example:
            >>> loader = DatasetLoader()
            >>> texts = loader.load_eval("sentiment_positive_eval")
        """
        file_path = self.eval_dir / f"{task_name}.txt"
        return self._load_file(file_path)

    def list_available(self, category: str = "all") -> dict:
        """List all available datasets.

        Args:
            category: Which category to list - "general", "tasks", "eval", or "all"

        Returns:
            Dictionary with available dataset names (without .txt extension)

        Example:
            >>> loader = DatasetLoader()
            >>> datasets = loader.list_available()
            >>> print(datasets["general"])
            ['mixed_domain', 'wikitext', 'coding', 'common_knowledge']
        """
        result = {}

        if category in ["general", "all"]:
            result["general"] = self._list_txt_files(self.general_dir)

        if category in ["tasks", "all"]:
            result["tasks"] = self._list_txt_files(self.tasks_dir)

        if category in ["eval", "all"]:
            result["eval"] = self._list_txt_files(self.eval_dir)

        return result

    def _list_txt_files(self, directory: Path) -> List[str]:
        """List all .txt files in a directory (without extension).

        Args:
            directory: Directory to search

        Returns:
            List of filenames without .txt extension
        """
        if not directory.exists():
            return []

        return sorted([
            f.stem for f in directory.glob("*.txt")
        ])

    def verify_setup(self) -> dict:
        """Verify that the data directory structure is set up correctly.

        Returns:
            Dictionary with status information and any issues found

        Example:
            >>> loader = DatasetLoader()
            >>> status = loader.verify_setup()
            >>> if status["ok"]:
            ...     print("Data directory is properly configured!")
        """
        issues = []

        # Check if directories exist
        if not self.data_dir.exists():
            issues.append(f"Data directory not found: {self.data_dir}")

        if not self.general_dir.exists():
            issues.append(f"General eval directory not found: {self.general_dir}")

        if not self.tasks_dir.exists():
            issues.append(f"Tasks directory not found: {self.tasks_dir}")

        if not self.eval_dir.exists():
            issues.append(f"Eval directory not found: {self.eval_dir}")

        # Get available datasets
        available = self.list_available()

        return {
            "ok": len(issues) == 0,
            "issues": issues,
            "available_datasets": available,
            "data_dir": str(self.data_dir.absolute()),
        }
