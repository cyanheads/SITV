"""
Task definitions for SITV experiments.

This module contains predefined tasks for multi-task experiments,
including sentiment analysis, instruction following, and QA tasks.
"""

from sitv.data.models import TaskDefinition


def get_predefined_tasks() -> dict[str, TaskDefinition]:
    """Get predefined task definitions for multi-task experiments.

    Returns:
        Dictionary mapping task names to TaskDefinition objects.
        Available tasks:
        - sentiment_positive: Positive sentiment analysis
        - sentiment_negative: Negative sentiment analysis
        - instruction_following: Following specific instruction formats
        - qa_factual: Factual question answering
    """

    tasks = {}

    # Task 1: Sentiment Analysis (Positive)
    tasks["sentiment_positive"] = TaskDefinition(
        name="sentiment_positive",
        description="Positive sentiment analysis",
        train_texts=[
            "This movie is absolutely fantastic! I loved every minute of it.",
            "The product exceeded all my expectations. Highly recommended!",
            "What a wonderful experience! I'm so happy with this purchase.",
            "Amazing quality and great value. I'm very satisfied.",
            "This is the best thing I've bought in years. Incredible!",
            "Outstanding service and excellent results. Very pleased!",
            "I'm delighted with this purchase. It works perfectly.",
            "Exceptional quality and superb performance. Love it!",
            "This is exactly what I needed. Fantastic product!",
            "Brilliant! This has made my life so much easier.",
        ] * 3,
        eval_texts=[
            "This product is amazing and works great!",
            "I'm very happy with my purchase. Excellent quality!",
            "Wonderful experience! Highly recommended.",
            "Best purchase I've made this year!",
            "Fantastic quality and great value.",
        ],
    )

    # Task 2: Sentiment Analysis (Negative)
    tasks["sentiment_negative"] = TaskDefinition(
        name="sentiment_negative",
        description="Negative sentiment analysis",
        train_texts=[
            "This movie was terrible. Complete waste of time.",
            "Very disappointed with this purchase. Poor quality.",
            "Worst product I've ever bought. Broke after one day.",
            "Horrible experience. Would not recommend to anyone.",
            "Absolutely awful quality. Total ripoff.",
            "Extremely dissatisfied. This doesn't work at all.",
            "Terrible service and poor product quality.",
            "Don't buy this! Save your money.",
            "Completely useless. Very frustrating experience.",
            "Worst decision I've made. Very unhappy.",
        ] * 3,
        eval_texts=[
            "This product is disappointing and doesn't work.",
            "Very unhappy with this purchase. Poor quality.",
            "Terrible experience! Do not recommend.",
            "Worst purchase I've made this year!",
            "Poor quality and bad value.",
        ],
    )

    # Task 3: Instruction Following
    tasks["instruction_following"] = TaskDefinition(
        name="instruction_following",
        description="Following specific instruction formats",
        train_texts=[
            "Instructions: List three colors.\nResponse: 1. Red 2. Blue 3. Green",
            "Instructions: Name two animals.\nResponse: 1. Cat 2. Dog",
            "Instructions: Give one example of a fruit.\nResponse: 1. Apple",
            "Instructions: State two countries.\nResponse: 1. France 2. Japan",
            "Instructions: List three numbers.\nResponse: 1. One 2. Two 3. Three",
            "Instructions: Name two programming languages.\nResponse: 1. Python 2. JavaScript",
            "Instructions: Give one day of the week.\nResponse: 1. Monday",
            "Instructions: List three planets.\nResponse: 1. Earth 2. Mars 3. Venus",
            "Instructions: Name two seasons.\nResponse: 1. Summer 2. Winter",
            "Instructions: Give one musical instrument.\nResponse: 1. Piano",
        ] * 3,
        eval_texts=[
            "Instructions: List three vegetables.\nResponse: ",
            "Instructions: Name two cities.\nResponse: ",
            "Instructions: Give one color.\nResponse: ",
            "Instructions: State two sports.\nResponse: ",
            "Instructions: List three tools.\nResponse: ",
        ],
    )

    # Task 4: Question Answering
    tasks["qa_factual"] = TaskDefinition(
        name="qa_factual",
        description="Factual question answering",
        train_texts=[
            "Q: What is the capital of France? A: The capital of France is Paris.",
            "Q: How many continents are there? A: There are seven continents.",
            "Q: What is the largest ocean? A: The Pacific Ocean is the largest ocean.",
            "Q: What year did World War II end? A: World War II ended in 1945.",
            "Q: What is the speed of light? A: The speed of light is approximately 299,792,458 meters per second.",
            "Q: Who invented the telephone? A: Alexander Graham Bell invented the telephone.",
            "Q: What is the tallest mountain? A: Mount Everest is the tallest mountain.",
            "Q: How many planets are in our solar system? A: There are eight planets in our solar system.",
            "Q: What is the chemical symbol for gold? A: The chemical symbol for gold is Au.",
            "Q: What is the largest country by area? A: Russia is the largest country by area.",
        ] * 3,
        eval_texts=[
            "Q: What is the capital of Germany? A:",
            "Q: How many days are in a year? A:",
            "Q: What is water made of? A:",
            "Q: Who painted the Mona Lisa? A:",
            "Q: What is the smallest prime number? A:",
        ],
    )

    return tasks
