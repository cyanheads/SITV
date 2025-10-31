"""Tests for dataset loader."""
import pytest
from pathlib import Path
from sitv.data.loader import DatasetLoader


class TestDatasetLoader:
    """Tests for DatasetLoader class."""

    @pytest.fixture
    def temp_data_dir(self, tmp_path):
        """Create temporary data directory structure for testing."""
        data_dir = tmp_path / "data"
        general_dir = data_dir / "general"
        tasks_dir = data_dir / "tasks"
        eval_dir = data_dir / "eval"

        general_dir.mkdir(parents=True)
        tasks_dir.mkdir(parents=True)
        eval_dir.mkdir(parents=True)

        return data_dir

    def test_initialization_with_custom_dir(self, temp_data_dir):
        """Should initialize with custom data directory."""
        loader = DatasetLoader(str(temp_data_dir))

        assert loader.data_dir == temp_data_dir
        assert loader.general_dir == temp_data_dir / "general"
        assert loader.tasks_dir == temp_data_dir / "tasks"
        assert loader.eval_dir == temp_data_dir / "eval"

    def test_initialization_default_dir(self):
        """Should initialize with default data directory."""
        loader = DatasetLoader()

        assert loader.data_dir.exists()
        assert loader.data_dir.name == "data"

    def test_load_file_basic(self, temp_data_dir):
        """Should load basic text file with examples."""
        test_file = temp_data_dir / "general" / "test.txt"
        test_file.write_text("""Example 1
Example 2
Example 3
""")

        loader = DatasetLoader(str(temp_data_dir))
        examples = loader._load_file(test_file)

        assert len(examples) == 3
        assert examples[0] == "Example 1"
        assert examples[1] == "Example 2"
        assert examples[2] == "Example 3"

    def test_load_file_with_comments(self, temp_data_dir):
        """Should skip comment lines starting with #."""
        test_file = temp_data_dir / "general" / "test.txt"
        test_file.write_text("""# This is a comment
Example 1
# Another comment
Example 2
""")

        loader = DatasetLoader(str(temp_data_dir))
        examples = loader._load_file(test_file)

        assert len(examples) == 2
        assert examples[0] == "Example 1"
        assert examples[1] == "Example 2"

    def test_load_file_with_empty_lines(self, temp_data_dir):
        """Should skip empty lines."""
        test_file = temp_data_dir / "general" / "test.txt"
        test_file.write_text("""Example 1

Example 2


Example 3
""")

        loader = DatasetLoader(str(temp_data_dir))
        examples = loader._load_file(test_file)

        assert len(examples) == 3

    def test_load_file_nonexistent_raises_error(self, temp_data_dir):
        """Should raise FileNotFoundError for missing file."""
        loader = DatasetLoader(str(temp_data_dir))

        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            loader._load_file(temp_data_dir / "nonexistent.txt")

    def test_load_file_empty_raises_error(self, temp_data_dir):
        """Should raise ValueError for empty file."""
        test_file = temp_data_dir / "general" / "empty.txt"
        test_file.write_text("")

        loader = DatasetLoader(str(temp_data_dir))

        with pytest.raises(ValueError, match="No examples found"):
            loader._load_file(test_file)

    def test_load_file_only_comments_raises_error(self, temp_data_dir):
        """Should raise ValueError for file with only comments."""
        test_file = temp_data_dir / "general" / "comments.txt"
        test_file.write_text("""# Comment 1
# Comment 2
# Comment 3
""")

        loader = DatasetLoader(str(temp_data_dir))

        with pytest.raises(ValueError, match="No examples found"):
            loader._load_file(test_file)

    def test_load_general(self, temp_data_dir):
        """Should load general evaluation dataset."""
        test_file = temp_data_dir / "general" / "mixed_domain.txt"
        test_file.write_text("""Sentence 1
Sentence 2
""")

        loader = DatasetLoader(str(temp_data_dir))
        examples = loader.load_general("mixed_domain")

        assert len(examples) == 2

    def test_load_general_combined(self, temp_data_dir):
        """Should load all general datasets when using 'combined'."""
        (temp_data_dir / "general" / "dataset1.txt").write_text("Line 1\nLine 2")
        (temp_data_dir / "general" / "dataset2.txt").write_text("Line 3\nLine 4")
        (temp_data_dir / "general" / "dataset3.txt").write_text("Line 5")

        loader = DatasetLoader(str(temp_data_dir))
        examples = loader.load_general("combined")

        assert len(examples) == 5
        assert "Line 1" in examples
        assert "Line 5" in examples

    def test_load_general_with_categories(self, temp_data_dir):
        """Should load general dataset with category labels."""
        test_file = temp_data_dir / "general" / "wikitext.txt"
        test_file.write_text("Wiki 1\nWiki 2\nWiki 3")

        loader = DatasetLoader(str(temp_data_dir))
        texts, categories = loader.load_general_with_categories("wikitext")

        assert len(texts) == 3
        assert len(categories) == 3
        assert all(c == "wikitext" for c in categories)

    def test_load_general_with_categories_combined(self, temp_data_dir):
        """Should load combined datasets with correct category labels."""
        (temp_data_dir / "general" / "coding.txt").write_text("Code 1\nCode 2")
        (temp_data_dir / "general" / "wikitext.txt").write_text("Wiki 1")

        loader = DatasetLoader(str(temp_data_dir))
        texts, categories = loader.load_general_with_categories("combined")

        assert len(texts) == 3
        assert len(categories) == 3

        # Check that categories match their texts
        coding_indices = [i for i, c in enumerate(categories) if c == "coding"]
        wiki_indices = [i for i, c in enumerate(categories) if c == "wikitext"]

        assert len(coding_indices) == 2
        assert len(wiki_indices) == 1

    def test_load_task(self, temp_data_dir):
        """Should load task training data."""
        test_file = temp_data_dir / "tasks" / "sentiment_positive.txt"
        test_file.write_text("Positive 1\nPositive 2")

        loader = DatasetLoader(str(temp_data_dir))
        examples = loader.load_task("sentiment_positive")

        assert len(examples) == 2

    def test_load_eval(self, temp_data_dir):
        """Should load task evaluation data."""
        test_file = temp_data_dir / "eval" / "sentiment_positive_eval.txt"
        test_file.write_text("Eval 1\nEval 2\nEval 3")

        loader = DatasetLoader(str(temp_data_dir))
        examples = loader.load_eval("sentiment_positive_eval")

        assert len(examples) == 3

    def test_list_available_all(self, temp_data_dir):
        """Should list all available datasets."""
        (temp_data_dir / "general" / "dataset1.txt").write_text("Line 1")
        (temp_data_dir / "general" / "dataset2.txt").write_text("Line 2")
        (temp_data_dir / "tasks" / "task1.txt").write_text("Task line")
        (temp_data_dir / "eval" / "eval1.txt").write_text("Eval line")

        loader = DatasetLoader(str(temp_data_dir))
        available = loader.list_available("all")

        assert "general" in available
        assert "tasks" in available
        assert "eval" in available
        assert len(available["general"]) == 2
        assert len(available["tasks"]) == 1
        assert len(available["eval"]) == 1

    def test_list_available_specific_category(self, temp_data_dir):
        """Should list datasets for specific category only."""
        (temp_data_dir / "general" / "dataset1.txt").write_text("Line 1")
        (temp_data_dir / "tasks" / "task1.txt").write_text("Task line")

        loader = DatasetLoader(str(temp_data_dir))
        available = loader.list_available("general")

        assert "general" in available
        assert "tasks" not in available
        assert len(available["general"]) == 1

    def test_list_txt_files_empty_directory(self, temp_data_dir):
        """Should return empty list for directory with no txt files."""
        loader = DatasetLoader(str(temp_data_dir))
        files = loader._list_txt_files(temp_data_dir / "general")

        assert files == []

    def test_list_txt_files_sorted(self, temp_data_dir):
        """Should return sorted list of txt file names."""
        (temp_data_dir / "general" / "zebra.txt").write_text("Z")
        (temp_data_dir / "general" / "apple.txt").write_text("A")
        (temp_data_dir / "general" / "monkey.txt").write_text("M")

        loader = DatasetLoader(str(temp_data_dir))
        files = loader._list_txt_files(temp_data_dir / "general")

        assert files == ["apple", "monkey", "zebra"]

    def test_verify_setup_success(self, temp_data_dir):
        """Should verify setup successfully when all directories exist."""
        loader = DatasetLoader(str(temp_data_dir))
        status = loader.verify_setup()

        assert status["ok"] is True
        assert len(status["issues"]) == 0
        assert "available_datasets" in status

    def test_verify_setup_missing_directories(self, tmp_path):
        """Should report issues when directories are missing."""
        loader = DatasetLoader(str(tmp_path / "nonexistent"))
        status = loader.verify_setup()

        assert status["ok"] is False
        assert len(status["issues"]) > 0

    def test_utf8_encoding(self, temp_data_dir):
        """Should handle UTF-8 encoded text correctly."""
        test_file = temp_data_dir / "general" / "unicode.txt"
        test_file.write_text("Hello 世界\nCafé ☕\n日本語")

        loader = DatasetLoader(str(temp_data_dir))
        examples = loader._load_file(test_file)

        assert len(examples) == 3
        assert "世界" in examples[0]
        assert "☕" in examples[1]
        assert "日本語" in examples[2]


class TestDatasetLoaderIntegration:
    """Integration tests using real data directory."""

    def test_load_real_datasets(self):
        """Should load real datasets from project data directory."""
        loader = DatasetLoader()

        # Verify data directory exists
        status = loader.verify_setup()
        if not status["ok"]:
            pytest.skip("Data directory not properly set up")

        # List available datasets
        available = loader.list_available("all")
        assert "general" in available
        assert "tasks" in available

        # Try to load a general dataset if available
        if available.get("general"):
            dataset_name = available["general"][0]
            texts = loader.load_general(dataset_name)
            assert len(texts) > 0

        # Try to load a task dataset if available
        if available.get("tasks"):
            task_name = available["tasks"][0]
            texts = loader.load_task(task_name)
            assert len(texts) > 0
