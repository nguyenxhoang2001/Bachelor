"""
Command-line interface for benchmark framework.
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="QCSP Benchmark Framework: MILP vs Heuristics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Part A (optimality check on small instances)
  python -m benchmark.cli run --config configs/benchmark.yaml --suite partA --output results/

  # Run Part B (primal bound race on larger instances)
  python -m benchmark.cli run --config configs/benchmark.yaml --suite partB --output results/

  # Run both suites
  python -m benchmark.cli run --config configs/benchmark.yaml --suite both --output results/

  # Generate plots
  python -m benchmark.cli plot --input results/ --suite both
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmark suite")
    run_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    run_parser.add_argument(
        "--suite",
        choices=["partA", "partB", "both"],
        required=True,
        help="Which suite to run"
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory (default: results/)"
    )
    
    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate plots from results")
    plot_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to results directory"
    )
    plot_parser.add_argument(
        "--suite",
        choices=["partA", "partB", "both"],
        required=True,
        help="Which suite to plot"
    )
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_benchmark(args)
    elif args.command == "plot":
        generate_plots(args)
    else:
        parser.print_help()


def run_benchmark(args):
    """Run benchmark suite."""
    from benchmark.run_partA import run_part_a
    from benchmark.run_partB import run_part_b
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.suite in ["partA", "both"]:
        print("\n" + "="*70)
        print("Running Part A: Optimality Check Suite")
        print("="*70 + "\n")
        run_part_a(args.config, str(output_dir))
    
    if args.suite in ["partB", "both"]:
        print("\n" + "="*70)
        print("Running Part B: Primal Bound Race Suite")
        print("="*70 + "\n")
        run_part_b(args.config, str(output_dir))


def generate_plots(args):
    """Generate plots from benchmark results."""
    from benchmark.plot_results import plot_part_a, plot_part_b
    
    input_path = Path(args.input)
    figures_dir = input_path / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if args.suite in ["partA", "both"]:
        print("\nGenerating Part A plots...")
        plot_part_a(str(input_path), str(figures_dir))
    
    if args.suite in ["partB", "both"]:
        print("\nGenerating Part B plots...")
        plot_part_b(str(input_path), str(figures_dir))
    
    print(f"\nAll plots saved to: {figures_dir}")


if __name__ == "__main__":
    main()
