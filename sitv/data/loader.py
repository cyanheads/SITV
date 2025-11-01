"""
Dataset loader for SITV experiments.

This module provides utilities for loading training and evaluation datasets
from text files in the data/ directory.
"""

from pathlib import Path
from typing import Dict, List, Tuple


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

    def __init__(self, data_dir: str | None = None):
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
                f"Dataset file not found: {file_path}\nExpected location: {file_path.absolute()}"
            )

        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # Strip whitespace
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                examples.append(line)

        if not examples:
            raise ValueError(
                f"No examples found in {file_path}. File may be empty or contain only comments."
            )

        return examples

    def load_general(self, dataset_name: str) -> List[str]:
        """Load a general evaluation dataset.

        Args:
            dataset_name: Name of the dataset (without .txt extension)
                         Options:
                         - "mixed_domain": Diverse multi-domain content
                         - "wikitext": Wikipedia-style factual content
                         - "coding": Programming and technical content
                         - "common_knowledge": Everyday general knowledge
                         - "combined": All general datasets combined (most comprehensive)

        Returns:
            List of evaluation text examples

        Example:
            >>> loader = DatasetLoader()
            >>> texts = loader.load_general("mixed_domain")
            >>> # Or load everything:
            >>> all_texts = loader.load_general("combined")
        """
        # Special case: combined loads all general datasets
        if dataset_name == "combined":
            all_texts = []
            available = self._list_txt_files(self.general_dir)
            for name in available:
                try:
                    texts = self._load_file(self.general_dir / f"{name}.txt")
                    all_texts.extend(texts)
                except Exception as e:
                    print(f"Warning: Failed to load {name}.txt: {e}")

            if not all_texts:
                raise ValueError(
                    "No general evaluation datasets found. "
                    "Expected at least one .txt file in data/general/"
                )

            return all_texts

        # Normal case: load specific dataset
        file_path = self.general_dir / f"{dataset_name}.txt"
        return self._load_file(file_path)

    def load_general_with_categories(self, dataset_name: str) -> Tuple[List[str], List[str]]:
        """Load general evaluation dataset with category labels.

        This method is useful for tracking which domain each example belongs to,
        enabling per-category loss analysis.

        Args:
            dataset_name: Name of the dataset (same options as load_general)

        Returns:
            Tuple of (texts, categories) where:
            - texts: List of text examples
            - categories: List of category labels (e.g., "coding", "wikitext")

        Example:
            >>> loader = DatasetLoader()
            >>> texts, categories = loader.load_general_with_categories("combined")
            >>> # texts[i] belongs to categories[i]
        """
        # Special case: combined loads all with labels
        if dataset_name == "combined":
            all_texts = []
            all_categories = []
            available = self._list_txt_files(self.general_dir)

            for name in available:
                try:
                    texts = self._load_file(self.general_dir / f"{name}.txt")
                    all_texts.extend(texts)
                    # Label each text with its source category
                    all_categories.extend([name] * len(texts))
                except FileNotFoundError:
                    print(f"Warning: Failed to load {name}.txt: File not found")
                except Exception as e:
                    print(f"Warning: Failed to load {name}.txt: {e}")

            if not all_texts:
                raise ValueError(
                    "No general evaluation datasets found. "
                    "Expected at least one .txt file in data/general/"
                )

            return all_texts, all_categories

        # Normal case: load specific dataset with uniform category
        file_path = self.general_dir / f"{dataset_name}.txt"
        texts = self._load_file(file_path)
        categories = [dataset_name] * len(texts)
        return texts, categories

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

        return sorted([f.stem for f in directory.glob("*.txt")])

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
