"""
Customizable Benchmark Script for QC Scheduling Problem

This script allows testing MILP vs Tabu search with configurable parameters:
- Number of QCs
- Time limit for MILP
- Number of test instances
- Instance size (small/medium/large)

Results are saved to CSV and plots showing primal bound development over time.
"""

import argparse
import csv
import os
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from qc_problem import QCProblem
from milp_solver import solve_milp
from tabu_solver import solve_tabu
from greedysolver import check_interference


# Instance size definitions
INSTANCE_SIZES = {
    "small": (3, 9),    # 3-9 tasks
    "medium": (10, 18), # 10-18 tasks
    "large": (19, 40),  # 19-40 tasks
}

FIXED_BAYS = 30


def parse_gurobi_log(log_file: str) -> List[Tuple[float, float]]:
    """
    Parse Gurobi log file to extract primal bound (incumbent) over time.
    
    Returns:
        List of (time, objective) tuples
    """
    primal_bounds = []
    
    if not os.path.exists(log_file):
        return primal_bounds
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Look for initial solution with format "Found heuristic solution: objective ..."
        if "Found heuristic solution: objective" in line:
            match = re.search(r'objective\s+([\d.e+-]+)', line)
            if match:
                obj = float(match.group(1))
                primal_bounds.append((0.0, obj))
        
        # Look for improved solutions marked with 'H' in the node log
        # Lines look like: "H  785   469                    55113.140436 9170.99203  83.4%  44.3    2s"
        # Format is: H [nodes] [unexplored] [incumbent] [bestbd] [gap%] [it/node] [time]
        if line.strip().startswith('H'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # Last part should be time with 's' suffix
                    time_str = parts[-1].replace('s', '')
                    t = float(time_str)
                    
                    # Incumbent is typically 5-7 positions from the end before gap%
                    # Search backwards from the percentage sign to find incumbent
                    # Gap appears as "XX.X%" so we look for that pattern
                    for i, part in enumerate(parts):
                        if '%' in part:
                            # Incumbent is typically 2 positions before the gap
                            if i >= 2:
                                try:
                                    obj = float(parts[i - 2])
                                    primal_bounds.append((t, obj))
                                    break
                                except ValueError:
                                    continue
                except (ValueError, IndexError):
                    continue
    
    return primal_bounds


def generate_instance(size_range: Tuple[int, int], num_qcs: int, seed: int) -> QCProblem:
    """Generate a random problem instance."""
    num_tasks = random.randint(size_range[0], size_range[1])
    
    problem = QCProblem.generate(
        num_bays=FIXED_BAYS,
        num_qcs=num_qcs,
        total_tasks=num_tasks,
        processing_time_range=(3.0, 180.0),
        travel_per_bay=1.0,
        precedence_probability=0.3,
        seed=seed
    )
    
    return problem


def run_milp_with_log(problem, time_limit: float, log_file: str) -> Dict:
    """Run MILP solver with logging enabled."""
    result = solve_milp(
        problem,
        time_limit=time_limit,
        verbose=False,
        log_file=log_file
    )
    return result


def run_tabu_with_trace(problem, max_time: float) -> Tuple[Dict[int, List[int]], float, List[Dict]]:
    """
    Run Tabu search and return best objective and trace.
    
    We run tabu for up to max_time seconds to match MILP time limit.
    """
    # Set high iteration limit so time is the limiting factor
    max_iterations = 100000
    
    schedule, objective, trace = solve_tabu(
        problem,
        max_iteration=max_iterations,
        max_time=max_time,
        tabu_tenure=None,
        tenure_jitter=3,
        move_types=("relocate", "swap", "intra_insert"),
        max_neighbors=None,
        seed=42,
        verbose=False,
        return_trace=True
    )
    
    return schedule, objective, trace


def plot_convergence(
    milp_bounds: List[Tuple[float, float]],
    tabu_trace: List[Dict],
    output_file: str,
    title: str,
    time_limit: float = None
):
    """
    Plot the primal bound development over time.
    
    Args:
        milp_bounds: List of (time, objective) for MILP
        tabu_trace: List of dicts with 'time' and 'best_objective'
        output_file: Path to save the plot
        title: Plot title
        time_limit: Time limit for extending lines to the right edge
    """
    plt.figure(figsize=(12, 7))
    
    # Determine maximum time for axis
    max_time = time_limit if time_limit else 0
    if milp_bounds:
        max_time = max(max_time, max(t for t, _ in milp_bounds))
    if tabu_trace:
        max_time = max(max_time, max(entry['time'] for entry in tabu_trace))
    
    # Plot MILP (red) first so it's behind
    if milp_bounds:
        # Sort by time, then by objective
        milp_bounds_sorted = sorted(milp_bounds, key=lambda x: (x[0], x[1]))
        
        # Create best-so-far progression
        best_objectives = []
        times_with_best = []
        current_best = float('inf')
        
        for t, obj in milp_bounds_sorted:
            if obj < current_best:
                if times_with_best:  # Add a point just before the improvement
                    times_with_best.append(t)
                    best_objectives.append(current_best)
                current_best = obj
                times_with_best.append(t)
                best_objectives.append(obj)
        
        # Extend to time_limit to show continued search with no improvement
        if times_with_best and max_time:
            if times_with_best[-1] < max_time:
                times_with_best.append(max_time)
                best_objectives.append(best_objectives[-1])
        
        if times_with_best:
            plt.step(times_with_best, best_objectives, 'r-', linewidth=2.5, label='MILP', where='post', alpha=0.8, marker='o', markersize=4)
    
    # Plot Tabu (blue) - already in best-so-far format
    if tabu_trace:
        tabu_times = [0.0] + [entry['time'] for entry in tabu_trace]
        tabu_objs = [tabu_trace[0]['best_objective']] + [entry['best_objective'] for entry in tabu_trace]
        
        # Extend to time_limit
        if tabu_times and max_time and tabu_times[-1] < max_time:
            tabu_times.append(max_time)
            tabu_objs.append(tabu_objs[-1])
        
        plt.step(tabu_times, tabu_objs, 'b-', linewidth=2.5, label='Tabu Search', where='post', alpha=0.75, marker='s', markersize=3)
    
    plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    plt.ylabel('Objective Value', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best', framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Fix y-axis formatting to avoid scientific notation offset
    ax = plt.gca()
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    
    plt.tight_layout()
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def run_benchmark(
    num_qcs: int,
    time_limit: float,
    num_instances: int,
    instance_size: str,
    output_dir: str = "benchmark"
):
    """
    Run the benchmark with specified configuration.
    
    Args:
        num_qcs: Number of quay cranes
        time_limit: Time limit for MILP in seconds
        num_instances: Number of test instances to generate
        instance_size: "small", "medium", or "large"
        output_dir: Directory to save results
    """
    # Validate instance size
    if instance_size not in INSTANCE_SIZES:
        raise ValueError(f"Invalid instance_size. Must be one of: {list(INSTANCE_SIZES.keys())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique run identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{num_qcs}qc_{instance_size}instance_{int(time_limit)}s_{timestamp}"
    
    csv_file = os.path.join(output_dir, f"{run_name}.csv")
    
    # Prepare CSV
    csv_rows = []
    
    print(f"\n{'='*70}")
    print(f"Starting Benchmark: {run_name}")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  - QCs: {num_qcs}")
    print(f"  - Time Limit: {time_limit}s")
    print(f"  - Instances: {num_instances}")
    print(f"  - Size: {instance_size} (tasks: {INSTANCE_SIZES[instance_size]})")
    print(f"  - Fixed Bays: {FIXED_BAYS}")
    print(f"{'='*70}\n")
    
    size_range = INSTANCE_SIZES[instance_size]
    
    for i in range(num_instances):
        print(f"[{i+1}/{num_instances}] Running instance {i+1}...")
        
        # Generate instance
        seed = random.randint(1, 1000000)
        problem = generate_instance(size_range, num_qcs, seed)
        num_tasks = len(problem.tasks)
        
        print(f"  Generated: {num_tasks} tasks, {num_qcs} QCs, {FIXED_BAYS} bays")
        
        # Run Tabu
        print(f"  Running Tabu Search...")
        tabu_start = time.time()
        tabu_schedule, tabu_objective, tabu_trace = run_tabu_with_trace(problem, time_limit)
        tabu_runtime = time.time() - tabu_start
        print(f"  Tabu completed: objective={tabu_objective:.2f}, runtime={tabu_runtime:.2f}s")

        tabu_ok, tabu_violation = check_interference(problem, tabu_schedule)
        if not tabu_ok:
            print(f"  Tabu schedule infeasible (interference): {tabu_violation}")
            tabu_objective = None
            tabu_trace = []
        
        # Run MILP with logging
        log_file = os.path.join(output_dir, f"{run_name}_inst{i+1}_milp.log")
        print(f"  Running MILP (time limit={time_limit}s)...")
        milp_start = time.time()
        milp_result = run_milp_with_log(problem, time_limit, log_file)
        milp_runtime = time.time() - milp_start
        
        milp_objective = milp_result.get('objective')
        milp_status = milp_result.get('status', 'UNKNOWN')
        
        print(f"  MILP completed: status={milp_status}, objective={milp_objective}, runtime={milp_runtime:.2f}s")
        
        # Calculate gap
        if milp_objective is not None and tabu_objective is not None:
            gap = abs(milp_objective - tabu_objective) / max(abs(tabu_objective), 1e-6) * 100
        else:
            gap = None
        
        # Parse MILP log for primal bounds
        print(f"  Parsing MILP log and generating plot...")
        milp_bounds = parse_gurobi_log(log_file)
        
        # Generate plot
        plot_file = os.path.join(output_dir, f"{run_name}_inst{i+1}.png")
        plot_title = f"Instance {i+1}: {num_qcs} QCs, {num_tasks} Tasks, {FIXED_BAYS} Bays"
        plot_convergence(milp_bounds, tabu_trace, plot_file, plot_title, time_limit=time_limit)
        print(f"  Plot saved: {plot_file}")
        
        # Store results
        csv_rows.append({
            "instance": i + 1,
            "num_qcs": num_qcs,
            "num_tasks": num_tasks,
            "num_bays": FIXED_BAYS,
            "seed": seed,
            "milp_objective": milp_objective if milp_objective is not None else "N/A",
            "milp_status": milp_status,
            "milp_runtime": f"{milp_runtime:.2f}",
            "tabu_objective": f"{tabu_objective:.2f}" if tabu_objective is not None else "N/A",
            "tabu_runtime": f"{tabu_runtime:.2f}",
            "gap_percent": f"{gap:.2f}" if gap is not None else "N/A",
        })
        
        print(f"  Instance {i+1} completed.\n")
    
    # Write CSV
    print(f"Writing results to {csv_file}...")
    with open(csv_file, 'w', newline='') as f:
        fieldnames = [
            "instance", "num_qcs", "num_tasks", "num_bays", "seed",
            "milp_objective", "milp_status", "milp_runtime",
            "tabu_objective", "tabu_runtime", "gap_percent"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"\n{'='*70}")
    print(f"Benchmark Complete!")
    print(f"Results saved to: {csv_file}")
    print(f"Plots saved to: {output_dir}/")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Customizable benchmark for QC scheduling problem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_custom.py --qcs 2 --time-limit 400 --instances 5 --size small
  python benchmark_custom.py --qcs 3 --time-limit 600 --instances 10 --size medium
  python benchmark_custom.py --qcs 4 --time-limit 1800 --instances 3 --size large
        """
    )
    
    parser.add_argument(
        '--qcs',
        type=int,
        required=True,
        help='Number of quay cranes (QCs)'
    )
    
    parser.add_argument(
        '--time-limit',
        type=float,
        required=True,
        help='Time limit for MILP solver in seconds'
    )
    
    parser.add_argument(
        '--instances',
        type=int,
        required=True,
        help='Number of test instances to generate and solve'
    )
    
    parser.add_argument(
        '--size',
        type=str,
        required=True,
        choices=['small', 'medium', 'large'],
        help='Instance size: small (3-9 tasks), medium (10-18 tasks), large (19+ tasks)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark',
        help='Output directory for results (default: benchmark)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Run benchmark
    run_benchmark(
        num_qcs=args.qcs,
        time_limit=args.time_limit,
        num_instances=args.instances,
        instance_size=args.size,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
