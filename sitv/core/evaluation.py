"""
Model evaluation service for SITV experiments.

This module provides the EvaluationService for evaluating language models
on text datasets.
"""

import torch
from torch import nn


class EvaluationService:
    """Service for evaluating language models.

    This service handles model evaluation on text datasets, computing
    perplexity and loss metrics. Supports batched evaluation and mixed
    precision for improved performance.

    Attributes:
        tokenizer: HuggingFace tokenizer for text preprocessing
        device: Device to run evaluation on
        batch_size: Number of texts to process per forward pass
        enable_mixed_precision: Whether to use FP16/BF16 for evaluation
        max_length: Maximum sequence length for tokenization
    """

    def __init__(
        self,
        tokenizer,
        device: str = "cuda",
        batch_size: int = 8,
        enable_mixed_precision: bool = True,
        max_length: int = 512
    ):
        """Initialize the evaluation service.

        Args:
            tokenizer: HuggingFace tokenizer
            device: Device for evaluation ("cuda", "mps", or "cpu")
            batch_size: Number of texts to evaluate per forward pass (default: 8)
            enable_mixed_precision: Use FP16/BF16 for faster evaluation (default: True)
            max_length: Maximum sequence length for tokenization (default: 512)
        """
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.enable_mixed_precision = enable_mixed_precision
        self.max_length = max_length

        # Determine dtype for mixed precision
        # BF16 is preferred on Ampere+ GPUs, FP16 on older CUDA or MPS
        self.autocast_dtype = None
        if enable_mixed_precision:
            if device == "cuda" and torch.cuda.is_available():
                # Check if BF16 is available (Ampere or newer)
                if torch.cuda.is_bf16_supported():
                    self.autocast_dtype = torch.bfloat16
                else:
                    self.autocast_dtype = torch.float16
            elif device == "mps":
                # MPS supports FP16
                self.autocast_dtype = torch.float16
            # CPU doesn't benefit from mixed precision, keep None

    def evaluate(self, model: nn.Module, texts: list[str]) -> float:
        """Evaluate model perplexity on evaluation texts.

        Uses batched evaluation and mixed precision for improved performance.
        Processes texts in batches to reduce forward pass overhead.

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
            # Process texts in batches for efficiency
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]

                # Tokenize batch with padding
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Use mixed precision if enabled
                if self.autocast_dtype is not None:
                    with torch.autocast(device_type=self.device, dtype=self.autocast_dtype):
                        outputs = model(**inputs, labels=inputs["input_ids"])
                else:
                    outputs = model(**inputs, labels=inputs["input_ids"])

                # Accumulate loss from batch
                # outputs.loss is already averaged over the batch
                total_loss += outputs.loss.item() * len(batch_texts)
                count += len(batch_texts)

        return total_loss / count if count > 0 else 0.0

    def evaluate_task_performance(
        self,
        model: nn.Module,
        task_eval_texts: list[str]
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
        texts: list[str]
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

    def evaluate_by_category(
        self,
        model: nn.Module,
        texts: list[str],
        categories: list[str]
    ) -> dict[str, float]:
        """Evaluate model separately for each category.

        This method groups texts by category and computes loss for each group,
        enabling analysis of how the model performs on different domains.

        Args:
            model: Model to evaluate
            texts: List of evaluation texts
            categories: List of category labels (same length as texts)

        Returns:
            Dictionary mapping category names to their average losses

        Example:
            >>> texts, categories = loader.load_general_with_categories("combined")
            >>> losses_by_cat = evaluator.evaluate_by_category(model, texts, categories)
            >>> print(f"Coding loss: {losses_by_cat['coding']:.4f}")
        """
        if len(texts) != len(categories):
            raise ValueError(
                f"texts and categories must have same length "
                f"(got {len(texts)} vs {len(categories)})"
            )

        # Group texts by category
        category_texts: dict[str, list[str]] = {}
        for text, category in zip(texts, categories, strict=True):
            if category not in category_texts:
                category_texts[category] = []
            category_texts[category].append(text)

        # Evaluate each category
        category_losses: dict[str, float] = {}
        for category, cat_texts in category_texts.items():
            loss = self.evaluate(model, cat_texts)
            category_losses[category] = loss

        return category_losses

    def evaluate_sentiment_preference(
        self,
        model: nn.Module,
        positive_texts: list[str],
        negative_texts: list[str]
    ) -> tuple[float, float, float]:
        """Evaluate model's sentiment preference.

        Computes loss on both positive and negative sentiment texts to measure
        whether the task vector has successfully learned a directional preference.

        Args:
            model: Model to evaluate
            positive_texts: List of positive sentiment examples
            negative_texts: List of negative sentiment examples

        Returns:
            Tuple of (positive_loss, negative_loss, preference_score)
            where preference_score = negative_loss - positive_loss
            (positive score means model prefers positive sentiment)

        Examples:
            >>> pos_loss, neg_loss, pref = evaluator.evaluate_sentiment_preference(
            ...     model, positive_examples, negative_examples
            ... )
            >>> print(f"Preference: {pref:+.4f} (>0 prefers positive)")
        """
        positive_loss = self.evaluate(model, positive_texts)
        negative_loss = self.evaluate(model, negative_texts)
        preference_score = negative_loss - positive_loss
        return positive_loss, negative_loss, preference_score
