"""
Model evaluation service for SITV experiments.

This module provides the EvaluationService for evaluating language models
on text datasets.
"""

import torch
import torch.nn as nn
from typing import List


class EvaluationService:
    """Service for evaluating language models.

    This service handles model evaluation on text datasets, computing
    perplexity and loss metrics.

    Attributes:
        tokenizer: HuggingFace tokenizer for text preprocessing
        device: Device to run evaluation on
    """

    def __init__(self, tokenizer, device: str = "cuda"):
        """Initialize the evaluation service.

        Args:
            tokenizer: HuggingFace tokenizer
            device: Device for evaluation ("cuda", "mps", or "cpu")
        """
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(self, model: nn.Module, texts: List[str]) -> float:
        """Evaluate model perplexity on evaluation texts.

        Args:
            model: Model to evaluate
            texts: List of evaluation texts

        Returns:
            Average loss across all texts (lower is better)

        Examples:
            >>> evaluator = EvaluationService(tokenizer, device="cuda")
            >>> loss = evaluator.evaluate(model, eval_texts)
            >>> perplexity = torch.exp(torch.tensor(loss)).item()
        """
        model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                count += 1

        return total_loss / count if count > 0 else 0.0

    def evaluate_task_performance(
        self,
        model: nn.Module,
        task_eval_texts: List[str]
    ) -> float:
        """Evaluate how well the model performs on a specific task.

        This is an alias for evaluate() for semantic clarity when
        evaluating task-specific performance.

        Args:
            model: Model to evaluate
            task_eval_texts: Task-specific evaluation texts

        Returns:
            Average loss on task texts (lower is better)

        Examples:
            >>> evaluator = EvaluationService(tokenizer, device="cuda")
            >>> task_loss = evaluator.evaluate_task_performance(model, task_texts)
        """
        return self.evaluate(model, task_eval_texts)

    def compute_perplexity(self, loss: float) -> float:
        """Convert loss to perplexity.

        Args:
            loss: Average cross-entropy loss

        Returns:
            Perplexity (exp(loss))

        Examples:
            >>> perplexity = evaluator.compute_perplexity(loss)
        """
        return torch.exp(torch.tensor(loss)).item()

    def evaluate_with_perplexity(
        self,
        model: nn.Module,
        texts: List[str]
    ) -> tuple[float, float]:
        """Evaluate model and return both loss and perplexity.

        Args:
            model: Model to evaluate
            texts: List of evaluation texts

        Returns:
            Tuple of (loss, perplexity)

        Examples:
            >>> loss, perplexity = evaluator.evaluate_with_perplexity(model, texts)
            >>> print(f"Loss: {loss:.4f}, Perplexity: {perplexity:.4f}")
        """
        loss = self.evaluate(model, texts)
        perplexity = self.compute_perplexity(loss)
        return loss, perplexity
