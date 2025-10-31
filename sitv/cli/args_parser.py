"""
Command-line argument parsing for SITV.

This module provides minimal argument parsing for the SITV CLI.
All configuration should be done via config.yaml.
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
  # Run full experiment using config.yaml
  python main.py

  # Run analysis only on previously fine-tuned models
  python main.py --analysis-only

  # Use custom config file
  python main.py --config my_config.yaml

Note: All experiment parameters should be configured in config.yaml.
The command-line interface is intentionally minimal to avoid configuration drift.
        """,
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file (default: ./config.yaml)",
    )

    # Operation mode
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Skip fine-tuning and run analysis on previously saved models",
    )

    return parser


def parse_arguments() -> Any:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace

    Examples:
        >>> args = parse_arguments()
    """
    parser = create_argument_parser()
    return parser.parse_args()
