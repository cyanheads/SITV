#!/usr/bin/env python3
"""
SITV - Self-Inverse Task Vectors
Loss Landscape Explorer using Refactored Architecture

This is the new thin entry point that uses the refactored modular architecture.
"""

from sitv.cli import parse_arguments
from sitv.experiments import ExperimentConfig, ExperimentOrchestrator


def main():
    """Main entry point for SITV experiments."""
    # Parse command-line arguments
    args = parse_arguments()

    # Create configuration from arguments
    config = ExperimentConfig.from_args(args)

    # Create and run orchestrator
    orchestrator = ExperimentOrchestrator(config)

    try:
        orchestrator.run()
    except NotImplementedError as e:
        print(f"\nError: {e}")
        print("\nThe refactored architecture is in place, but fine-tuning")
        print("logic still needs to be migrated from original main.py.")
        print("\nTo use the refactored code:")
        print("1. Fine-tune models using original main.py")
        print("2. Run analysis with: python main_new.py --analysis-only")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
