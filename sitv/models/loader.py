"""
Model loading and saving service for SITV.

This module provides the ModelService for loading, saving, and managing
HuggingFace transformer models.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from typing import Tuple, Optional


class ModelService:
    """Service for model lifecycle management.

    This service handles:
    - Loading models from HuggingFace
    - Saving models to disk
    - Loading saved models
    - Checking for existing saved models
    - Managing device placement

    Attributes:
        device_map: Device map for model loading
        torch_dtype: Data type for model parameters
    """

    def __init__(
        self,
        device_map: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        """Initialize the model service.

        Args:
            device_map: Device map for distributed loading (None or "auto")
            torch_dtype: Data type for model parameters
        """
        self.device_map = device_map
        self.torch_dtype = torch_dtype

    def load_model(
        self,
        model_name: str,
        device: Optional[str] = None
    ) -> PreTrainedModel:
        """Load a model from HuggingFace.

        Args:
            model_name: HuggingFace model identifier
            device: Optional device to move model to (if device_map is None)

        Returns:
            Loaded model

        Examples:
            >>> service = ModelService()
            >>> model = service.load_model("gpt2")
        """
        print(f"Loading model: {model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map
        )

        # Move to device if specified and no device_map
        if device and self.device_map is None:
            model = model.to(device)  # type: ignore[assignment]

        print(f"Model loaded successfully")
        return model

    def load_tokenizer(self, model_name: str):
        """Load a tokenizer from HuggingFace.

        Args:
            model_name: HuggingFace model identifier

        Returns:
            Loaded tokenizer

        Examples:
            >>> service = ModelService()
            >>> tokenizer = service.load_tokenizer("gpt2")
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def load_model_and_tokenizer(
        self,
        model_name: str,
        device: Optional[str] = None
    ) -> Tuple[PreTrainedModel, AutoTokenizer]:
        """Load both model and tokenizer.

        Args:
            model_name: HuggingFace model identifier
            device: Optional device to move model to

        Returns:
            Tuple of (model, tokenizer)

        Examples:
            >>> service = ModelService()
            >>> model, tokenizer = service.load_model_and_tokenizer("gpt2")
        """
        model = self.load_model(model_name, device)
        tokenizer = self.load_tokenizer(model_name)
        return model, tokenizer

    def save_models(
        self,
        base_model: PreTrainedModel,
        finetuned_model: PreTrainedModel,
        output_dir: str
    ) -> None:
        """Save base and fine-tuned models for later analysis.

        Args:
            base_model: Base model to save
            finetuned_model: Fine-tuned model to save
            output_dir: Directory to save models

        Examples:
            >>> service = ModelService()
            >>> service.save_models(base_model, finetuned_model, "outputs")
        """
        base_path = os.path.join(output_dir, "saved_base_model")
        ft_path = os.path.join(output_dir, "saved_finetuned_model")

        print(f"\nSaving models for future analysis...")
        print(f"  Base model → {base_path}")
        print(f"  Fine-tuned model → {ft_path}")

        # Save models (saves config, weights, etc.)
        base_model.save_pretrained(base_path)
        finetuned_model.save_pretrained(ft_path)

        print("Models saved successfully!\n")

    def load_saved_models(
        self,
        output_dir: str,
        device: str
    ) -> Tuple[PreTrainedModel, PreTrainedModel]:
        """Load previously saved models for analysis.

        Args:
            output_dir: Directory containing saved models
            device: Device to load models to

        Returns:
            Tuple of (base_model, finetuned_model)

        Examples:
            >>> service = ModelService()
            >>> base, finetuned = service.load_saved_models("outputs", "cuda")
        """
        base_path = os.path.join(output_dir, "saved_base_model")
        ft_path = os.path.join(output_dir, "saved_finetuned_model")

        print(f"\nLoading previously saved models...")
        print(f"  Base model ← {base_path}")
        print(f"  Fine-tuned model ← {ft_path}")

        # Load models
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=self.torch_dtype,
            device_map=None
        )
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            ft_path,
            torch_dtype=self.torch_dtype,
            device_map=None
        )

        # Move to device
        base_model = base_model.to(device)  # type: ignore[assignment]
        finetuned_model = finetuned_model.to(device)  # type: ignore[assignment]

        print("Models loaded successfully!\n")

        return base_model, finetuned_model

    @staticmethod
    def check_saved_models_exist(output_dir: str) -> bool:
        """Check if saved models exist in the output directory.

        Args:
            output_dir: Directory to check for saved models

        Returns:
            True if both base and fine-tuned models exist, False otherwise

        Examples:
            >>> if ModelService.check_saved_models_exist("outputs"):
            ...     print("Saved models found!")
        """
        base_path = os.path.join(output_dir, "saved_base_model")
        ft_path = os.path.join(output_dir, "saved_finetuned_model")

        base_exists = os.path.exists(os.path.join(base_path, "config.json"))
        ft_exists = os.path.exists(os.path.join(ft_path, "config.json"))

        return base_exists and ft_exists

    @staticmethod
    def count_parameters(model: PreTrainedModel) -> int:
        """Count the total number of parameters in a model.

        Args:
            model: Model to count parameters for

        Returns:
            Total number of parameters

        Examples:
            >>> param_count = ModelService.count_parameters(model)
            >>> print(f"Model has {param_count:,} parameters")
        """
        return sum(p.numel() for p in model.parameters())
