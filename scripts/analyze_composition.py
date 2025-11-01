#!/usr/bin/env python3
"""
Analyze 2D task vector composition.

This is a convenience wrapper around sitv.analysis.CompositionAnalyzer.
For library usage, import directly from sitv.analysis.

Usage:
    python scripts/analyze_composition.py <data_dir>
    python scripts/analyze_composition.py findings/2025-11-01/T1
"""

import sys
from pathlib import Path


def main():
    """Run composition analysis from command line."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_composition.py <data_dir>")
        print("\nExample:")
        print("  python scripts/analyze_composition.py findings/2025-11-01/T1")
        sys.exit(1)

    # Import here to give better error message if module not installed
    try:
        from sitv.analysis import CompositionAnalyzer
    except ImportError:
        print("Error: SITV package not found.")
        print("Install with: pip install -e .")
        sys.exit(1)

    data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    # Run analysis
    analyzer = CompositionAnalyzer(data_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
