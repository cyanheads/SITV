"""Tests for model management functions."""
import pytest
import os
from main import check_saved_models_exist


class TestCheckSavedModelsExist:
    """Tests for check_saved_models_exist function."""

    def test_no_models_exist(self, temp_output_dir):
        """Should return False when no models exist."""
        assert check_saved_models_exist(temp_output_dir) is False

    def test_only_base_model_exists(self, temp_output_dir):
        """Should return False when only base model exists."""
        base_path = os.path.join(temp_output_dir, "saved_base_model")
        os.makedirs(base_path, exist_ok=True)

        # Create config.json
        config_path = os.path.join(base_path, "config.json")
        with open(config_path, "w") as f:
            f.write("{}")

        assert check_saved_models_exist(temp_output_dir) is False

    def test_only_finetuned_model_exists(self, temp_output_dir):
        """Should return False when only finetuned model exists."""
        ft_path = os.path.join(temp_output_dir, "saved_finetuned_model")
        os.makedirs(ft_path, exist_ok=True)

        # Create config.json
        config_path = os.path.join(ft_path, "config.json")
        with open(config_path, "w") as f:
            f.write("{}")

        assert check_saved_models_exist(temp_output_dir) is False

    def test_both_models_exist(self, temp_output_dir):
        """Should return True when both models exist."""
        # Create base model directory and config
        base_path = os.path.join(temp_output_dir, "saved_base_model")
        os.makedirs(base_path, exist_ok=True)
        with open(os.path.join(base_path, "config.json"), "w") as f:
            f.write("{}")

        # Create finetuned model directory and config
        ft_path = os.path.join(temp_output_dir, "saved_finetuned_model")
        os.makedirs(ft_path, exist_ok=True)
        with open(os.path.join(ft_path, "config.json"), "w") as f:
            f.write("{}")

        assert check_saved_models_exist(temp_output_dir) is True

    def test_directories_exist_but_no_configs(self, temp_output_dir):
        """Should return False if directories exist but config.json is missing."""
        base_path = os.path.join(temp_output_dir, "saved_base_model")
        ft_path = os.path.join(temp_output_dir, "saved_finetuned_model")
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(ft_path, exist_ok=True)

        # No config.json files created
        assert check_saved_models_exist(temp_output_dir) is False
