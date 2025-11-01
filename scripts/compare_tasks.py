#!/usr/bin/env python3
"""
Multi-Task Loss Landscape Comparison Script

Loads results from multiple task experiments (SP, SN, QA) and creates
combined visualizations to compare loss landscapes across different tasks.

Usage:
    python scripts/compare_tasks.py
    python scripts/compare_tasks.py --findings-dir findings/2025-10-31
    python scripts/compare_tasks.py --layouts heatmap
    python scripts/compare_tasks.py --layouts publication overlaid
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
from sitv.data.models import AlphaSweepResult
from sitv.reporting import ComparisonReportGenerator
from sitv.visualization.plotter import ResultPlotter


def load_task_data(task_dir: Path) -> Dict[str, Any]:
    """Load results and analysis from a task directory.

    Args:
        task_dir: Path to task directory containing JSON files

    Returns:
        Dictionary with 'results' and 'analysis' keys
    """
    results_path = task_dir / "loss_landscape_results.json"
    analysis_path = task_dir / "analysis_results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    if not analysis_path.exists():
        raise FileNotFoundError(f"Analysis file not found: {analysis_path}")

    with open(results_path, 'r') as f:
        results_json = json.load(f)

    with open(analysis_path, 'r') as f:
        analysis_json = json.load(f)

    # Convert to AlphaSweepResult objects using dataclass constructor
    results = [AlphaSweepResult(**r) for r in results_json]

    analysis = {
        'min_general_loss': AlphaSweepResult(**analysis_json['min_general_loss']),
        'min_task_loss': AlphaSweepResult(**analysis_json['min_task_loss']),
        'best_return': AlphaSweepResult(**analysis_json['best_return']),
        'zero_crossings': [AlphaSweepResult(**r) for r in analysis_json.get('zero_crossings', [])],
        'sorted_by_return': [AlphaSweepResult(**r) for r in analysis_json.get('sorted_by_return', [])],
        'squaring_return_points': [AlphaSweepResult(**r) for r in analysis_json.get('squaring_return_points', [])],
        'has_squaring_data': analysis_json.get('has_squaring_data', False)
    }

    return {'results': results, 'analysis': analysis}




def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate combined visualizations from multiple task experiments"
    )
    parser.add_argument(
        '--findings-dir',
        type=str,
        default='findings/2025-10-31',
        help='Directory containing task subdirectories (default: findings/2025-10-31)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save combined outputs (default: <findings-dir>/combined_analysis)'
    )
    parser.add_argument(
        '--layouts',
        nargs='+',
        choices=['overlaid', 'side_by_side', 'grid', 'publication', 'heatmap', 'all'],
        default=['all'],
        help='Which visualization layouts to generate (default: all)'
    )

    args = parser.parse_args()

    # Setup paths
    findings_dir = Path(args.findings_dir)
    output_dir = Path(args.output_dir) if args.output_dir else findings_dir / "combined_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔍 Loading task data from: {findings_dir}")

    # Load data from each task directory
    task_datasets = {}
    task_dirs = {
        'SP': 'SP',  # Sentiment Positive
        'SN': 'SN',  # Sentiment Negative
        'QA': 'QA',  # Question Answering / Instruction Following
    }

    for dir_name, display_name in task_dirs.items():
        task_dir = findings_dir / dir_name
        if task_dir.exists():
            print(f"  ✓ Loading {display_name} from {task_dir}")
            try:
                task_datasets[display_name] = load_task_data(task_dir)
            except Exception as e:
                print(f"  ⚠️  Error loading {display_name}: {e}")
        else:
            print(f"  ⚠️  Skipping {display_name} (directory not found: {task_dir})")

    if not task_datasets:
        print("\n❌ Error: No task data found!")
        return 1

    print(f"\n✓ Loaded {len(task_datasets)} tasks: {', '.join(task_datasets.keys())}")

    # Create plotter
    plotter = ResultPlotter(figsize=(16, 10), dpi=100)

    # Determine layouts
    if 'all' in args.layouts:
        layouts = ['heatmap', 'overlaid', 'side_by_side', 'grid', 'publication']
    else:
        layouts = args.layouts

    print(f"\n📊 Generating {len(layouts)} visualization(s)...")

    # Generate visualizations
    for layout in layouts:
        output_path = output_dir / f"combined_{layout}.png"
        print(f"\n  Creating {layout} layout...")
        plotter.plot_multi_task_comparison(
            task_datasets,
            output_path=str(output_path),
            layout=layout
        )

    # Generate comparison report
    print("\n📄 Generating comparison report...")
    report_path = output_dir / "comparison_report.md"
    report_generator = ComparisonReportGenerator()
    report_generator.generate(task_datasets, report_path)

    # Save combined data as JSON (using vars() to convert dataclasses)
    print("\n💾 Saving combined data...")
    combined_json = {}
    for task_name, data in task_datasets.items():
        combined_json[task_name] = {
            'results': [vars(r) for r in data['results']],
            'analysis': {
                'min_general_loss': vars(data['analysis']['min_general_loss']),
                'min_task_loss': vars(data['analysis']['min_task_loss']),
                'best_return': vars(data['analysis']['best_return']),
                'zero_crossings': [vars(r) for r in data['analysis']['zero_crossings']],
                'num_zero_crossings': len(data['analysis']['zero_crossings']),
                'squaring_return_points': [vars(r) for r in data['analysis']['squaring_return_points']],
                'num_squaring_returns': len(data['analysis']['squaring_return_points']),
                'has_squaring_data': data['analysis']['has_squaring_data']
            }
        }

    combined_json_path = output_dir / "combined_data.json"
    with open(combined_json_path, 'w') as f:
        json.dump(combined_json, f, indent=2)

    print(f"  ✓ Combined data saved to {combined_json_path}")

    print("\n✅ Analysis complete!")
    print(f"\n📁 All outputs saved to: {output_dir}/")
    print(f"  - {len(layouts)} PNG visualization(s)")
    print(f"  - 1 comparison report (comparison_report.md)")
    print(f"  - 1 combined data file (combined_data.json)")

    return 0


if __name__ == '__main__':
    exit(main())
