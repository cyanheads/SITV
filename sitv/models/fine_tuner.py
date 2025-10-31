"""
Fine-tuning service for SITV.

This module provides the FineTuner service for fine-tuning language models
on specific tasks.
"""

import time
from datetime import datetime
from typing import Dict, List, Any, Tuple
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from sitv.utils.progress import FineTuningProgressCallback


class FineTuner:
    """Service for fine-tuning language models.

    This class encapsulates all fine-tuning logic including:
    - Dataset preparation
    - Training configuration
    - Progress tracking
    - Model training

    Attributes:
        output_dir: Directory for training outputs/checkpoints
        num_epochs: Number of training epochs
        learning_rate: Learning rate for fine-tuning
        batch_size: Training batch size
        max_length: Maximum sequence length for tokenization
        save_strategy: Model save strategy
        logging_steps: Steps between logging

    Examples:
        >>> tuner = FineTuner(num_epochs=3, learning_rate=5e-5)
        >>> finetuned_model, metrics = tuner.fine_tune(
        ...     base_model=model,
        ...     tokenizer=tokenizer,
        ...     train_texts=train_data
        ... )
    """

    def __init__(
        self,
        output_dir: str = "./finetuned_model",
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        batch_size: int = 1,
        max_length: int = 512,
        save_strategy: str = "no",
        logging_steps: int = 10,
        seed: int = 42,
    ):
        """Initialize the fine-tuner.

        Args:
            output_dir: Directory for training outputs
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Training batch size
            max_length: Maximum sequence length for tokenization
            save_strategy: Model save strategy ("no", "steps", "epoch")
            logging_steps: Number of steps between logging
            seed: Random seed for data shuffling and sampling
        """
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.save_strategy = save_strategy
        self.logging_steps = logging_steps
        self.seed = seed

    def fine_tune(
        self,
        base_model: PreTrainedModel,
        tokenizer,
        train_texts: List[str],
    ) -> Tuple[PreTrainedModel, Dict[str, Any]]:
        """Fine-tune model on task to create a structured task vector.

        Args:
            base_model: Pre-trained base model
            tokenizer: Associated tokenizer
            train_texts: Training examples for the task

        Returns:
            Tuple of (Fine-tuned model, metrics dict)

        Examples:
            >>> tuner = FineTuner()
            >>> finetuned_model, metrics = tuner.fine_tune(
            ...     base_model=model,
            ...     tokenizer=tokenizer,
            ...     train_texts=["example 1", "example 2"]
            ... )
        """
        ft_start_time = time.time()

        print("\n" + "="*70)
        print("FINE-TUNING MODEL")
        print("="*70)
        print(f"  Training examples: {len(train_texts)}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Learning rate: {self.learning_rate:.2e}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Max length: {self.max_length}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Prepare dataset
        dataset = self._prepare_dataset(train_texts, tokenizer)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            save_strategy=self.save_strategy,
            logging_steps=self.logging_steps,
            seed=self.seed,  # Ensure reproducible data sampling
            report_to=[],  # Disable wandb/tensorboard
            remove_unused_columns=False,
            gradient_checkpointing=True,  # Save memory for large models
            fp16=False,  # Using bfloat16 instead
            bf16=True,  # Match model's native precision
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False  # Causal LM, not masked LM
        )

        # Progress callback
        progress_callback = FineTuningProgressCallback()

        # Trainer
        trainer = Trainer(
            model=base_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            callbacks=[progress_callback],
        )

        # Train
        train_result = trainer.train()

        ft_end_time = time.time()
        ft_duration = ft_end_time - ft_start_time

        # Collect metrics
        metrics = {
            "start_time": datetime.fromtimestamp(ft_start_time).isoformat(),
            "end_time": datetime.fromtimestamp(ft_end_time).isoformat(),
            "duration_seconds": ft_duration,
            "training_examples": len(train_texts),
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "final_loss": train_result.training_loss,
            "training_steps": train_result.global_step,
            "training_history": progress_callback.training_history,
        }

        print(f"\n{'='*70}")
        print("FINE-TUNING COMPLETE")
        print(f"{'='*70}")
        print(f"  Duration: {ft_duration / 60:.1f} minutes ({ft_duration:.0f}s)")
        print(f"  Final loss: {train_result.training_loss:.4f}")
        print(f"  Total steps: {train_result.global_step}")
        print(f"  Avg time/step: {ft_duration / train_result.global_step:.2f}s")
        print(f"{'='*70}\n")

        return trainer.model, metrics

    def _prepare_dataset(
        self,
        texts: List[str],
        tokenizer,
    ) -> Dataset:
        """Prepare dataset for training.

        Args:
            texts: List of training texts
            tokenizer: Tokenizer for encoding

        Returns:
            Dataset object ready for training
        """
        return TextDataset(texts, tokenizer, self.max_length)


class TextDataset(Dataset):
    """Dataset wrapper for tokenizing text inputs for language model training.

    This dataset handles tokenization and padding for text inputs.

    Attributes:
        encodings: Tokenized and encoded texts

    Examples:
        >>> dataset = TextDataset(
        ...     texts=["text 1", "text 2"],
        ...     tokenizer=tokenizer,
        ...     max_length=512
        ... )
        >>> dataset[0]
        {'input_ids': tensor([...]), 'attention_mask': tensor([...])}
    """

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        """Initialize the text dataset.

        Args:
            texts: List of text strings to tokenize
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of examples in the dataset
        """
        return len(self.encodings.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example.

        Args:
            idx: Index of the example

        Returns:
            Dictionary with input_ids and attention_mask
        """
        return {
            "input_ids": self.encodings.input_ids[idx],
            "attention_mask": self.encodings.attention_mask[idx],
        }
