"""
Command-line argument parsing for SITV.

This module provides argument parsing for the SITV command-line interface.
"""

import argparse
from typing import Any


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser for SITV.

    Returns:
        Configured ArgumentParser object

    Examples:
        >>> parser = create_argument_parser()
        >>> args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(
        description="Task Vector Loss Landscape Explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full experiment (fine-tuning + analysis)
  python main.py

  # Run analysis only on previously fine-tuned models
  python main.py --analysis-only

  # Run with custom alpha range and samples
  python main.py --alpha-min -5 --alpha-max 5 --num-samples 200

  # Enable 2D composition experiment
  python main.py --enable-2d-composition

  # Run specific task
  python main.py --task sentiment_negative

  # Analysis only with custom parameters
  python main.py --analysis-only --alpha-min -2 --alpha-max 2 --num-samples 50
        """,
    )

    # Operation mode
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip fine-tuning and run analysis on previously saved models",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model identifier (default: Qwen/Qwen2.5-0.5B)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for computation (cuda/mps/cpu, default: auto-detect)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory (default: ./outputs)",
    )

    # Alpha sweep configuration
    parser.add_argument(
        "--alpha-min",
        type=float,
        default=-3.0,
        help="Minimum alpha value for sweep (default: -3.0)",
    )

    parser.add_argument(
        "--alpha-max",
        type=float,
        default=3.0,
        help="Maximum alpha value for sweep (default: 3.0)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of alpha samples (default: 100)",
    )

    # Squaring test
    parser.add_argument(
        "--enable-squaring-test",
        action="store_true",
        default=True,
        help="Enable squaring test: evaluate M(2α) to test [W(λ)]² = I analog (default: enabled)",
    )

    parser.add_argument(
        "--disable-squaring-test",
        action="store_true",
        help="Disable squaring test to speed up sweep",
    )

    # Task configuration
    parser.add_argument(
        "--task",
        type=str,
        default="sentiment_positive",
        choices=["sentiment_positive", "sentiment_negative", "instruction_following", "qa_factual"],
        help="Task to use for fine-tuning (default: sentiment_positive)",
    )

    parser.add_argument(
        "--multi-task",
        action="store_true",
        help="Run experiment on all available tasks for comparison",
    )

    # 2D composition experiment
    parser.add_argument(
        "--enable-2d-composition",
        "--enable-2d",
        action="store_true",
        dest="enable_2d",
        help="Enable 2D task vector composition experiment M(α,β) = M_base + α·T1 + β·T2",
    )

    parser.add_argument(
        "--samples-2d",
        type=int,
        default=20,
        help="Samples per dimension for 2D composition (creates NxN grid, default: 20)",
    )

    parser.add_argument(
        "--alpha-2d-min",
        type=float,
        default=-2.0,
        help="Minimum alpha value for 2D composition (default: -2.0)",
    )

    parser.add_argument(
        "--alpha-2d-max",
        type=float,
        default=2.0,
        help="Maximum alpha value for 2D composition (default: 2.0)",
    )

    parser.add_argument(
        "--beta-min",
        type=float,
        default=-2.0,
        help="Minimum beta value for 2D composition (default: -2.0)",
    )

    parser.add_argument(
        "--beta-max",
        type=float,
        default=2.0,
        help="Maximum beta value for 2D composition (default: 2.0)",
    )

    # Fine-tuning configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs (default: 3)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning (default: 1e-4)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for fine-tuning (default: 4)",
    )

    return parser


def parse_arguments() -> Any:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace

    Examples:
        >>> args = parse_arguments()
        >>> print(args.model)
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Handle squaring test flag logic
    args.enable_squaring = args.enable_squaring_test and not args.disable_squaring_test

    return args
