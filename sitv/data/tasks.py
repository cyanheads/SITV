"""
Task definitions for SITV experiments.

This module contains predefined tasks for multi-task experiments,
including sentiment analysis, instruction following, and QA tasks.

Tasks are now loaded from text files in the data/ directory.
"""

from sitv.data.loader import DatasetLoader
from sitv.data.models import TaskDefinition


def get_predefined_tasks(data_repetition_factor: int = 100) -> dict[str, TaskDefinition]:
    """Get predefined task definitions for multi-task experiments.

    Tasks are loaded from text files in the data/ directory and training
    examples are repeated according to data_repetition_factor.

    Args:
        data_repetition_factor: Multiplier for repeating training examples.
            Each task has ~30 unique examples that will be repeated this many times.
            Default 100 gives 3000 training examples per task.

    Returns:
        Dictionary mapping task names to TaskDefinition objects.
        Available tasks:
        - sentiment_positive: Positive sentiment analysis
        - sentiment_negative: Negative sentiment analysis
        - instruction_following: Following specific instruction formats
        - qa_factual: Factual question answering
    """

    # Initialize the data loader
    loader = DatasetLoader()

    tasks = {}

    # Task 1: Sentiment Analysis (Positive)
    train_texts = loader.load_task("sentiment_positive")
    eval_texts = loader.load_eval("sentiment_positive_eval")

    tasks["sentiment_positive"] = TaskDefinition(
        name="sentiment_positive",
        description="Positive sentiment analysis",
        train_texts=train_texts * data_repetition_factor,
        eval_texts=eval_texts,
    )

    # Task 2: Sentiment Analysis (Negative)
    train_texts = loader.load_task("sentiment_negative")
    eval_texts = loader.load_eval("sentiment_negative_eval")

    tasks["sentiment_negative"] = TaskDefinition(
        name="sentiment_negative",
        description="Negative sentiment analysis",
        train_texts=train_texts * data_repetition_factor,
        eval_texts=eval_texts,
    )

    # Task 3: Instruction Following
    train_texts = loader.load_task("instruction_following")
    eval_texts = loader.load_eval("instruction_following_eval")

    tasks["instruction_following"] = TaskDefinition(
        name="instruction_following",
        description="Following specific instruction formats",
        train_texts=train_texts * data_repetition_factor,
        eval_texts=eval_texts,
    )

    # Task 4: Question Answering
    train_texts = loader.load_task("qa_factual")
    eval_texts = loader.load_eval("qa_factual_eval")

    tasks["qa_factual"] = TaskDefinition(
        name="qa_factual",
        description="Factual question answering",
        train_texts=train_texts * data_repetition_factor,
        eval_texts=eval_texts,
    )

    return tasks
