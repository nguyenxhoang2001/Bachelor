"""
Plotting module for benchmark visualization.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import csv
from collections import defaultdict
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def load_csv(filepath: str) -> List[Dict]:
    """Load CSV file into list of dicts."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_part_a(input_dir: str, figures_dir: str):
    """Generate plots for Part A results."""
    input_path = Path(input_dir)
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_file = input_path / "partA_results.csv"
    if not results_file.exists():
        print(f"Part A results not found: {results_file}")
        return
    
    results = load_csv(str(results_file))
    
    # Group by configuration
    by_config = defaultdict(list)
    for row in results:
        config_id = row["config_id"]
        by_config[config_id].append(row)
    
    # Plot 1: Optimality match rate by task size
    plot_optimality_match_rate(by_config, str(figures_path / "partA_optimality_match_rate.png"))
    
    # Plot 2: Gap distribution
    plot_gap_distribution(by_config, str(figures_path / "partA_gap_distribution.png"))
    
    # Plot 3: Runtime comparison
    plot_runtime_comparison(by_config, str(figures_path / "partA_runtime_comparison.png"))
    
    print(f"Part A plots saved to {figures_path}")


def plot_optimality_match_rate(by_config: Dict, output_file: str):
    """Bar plot of optimality match rate by task size."""
    # Extract data
    configs = sorted(by_config.keys())
    
    # Group by task size and QC count
    by_tasks = defaultdict(lambda: defaultdict(list))
    for config_id, rows in by_config.items():
        num_tasks = rows[0]["num_tasks"]
        num_qcs = rows[0]["num_qcs"]
        matches = sum(1 for r in rows if r["is_optimal_match"] == True or r["is_optimal_match"] == "True")
        rate = matches / len(rows) * 100 if rows else 0
        by_tasks[num_tasks][num_qcs].append(rate)
    
    # Prepare plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    task_sizes = sorted(by_tasks.keys())
    qc_counts = sorted(set(qc for tasks in by_tasks.values() for qc in tasks.keys()))
    
    x = np.arange(len(task_sizes))
    width = 0.35
    
    for i, qc in enumerate(qc_counts):
        rates = [np.mean(by_tasks[t].get(qc, [0])) for t in task_sizes]
        ax.bar(x + i * width, rates, width, label=f'{qc} QCs', alpha=0.8)
    
    ax.set_xlabel('Number of Tasks', fontsize=12, fontweight='bold')
    ax.set_ylabel('Optimality Match Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Heuristic Optimality Match Rate by Instance Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(task_sizes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_gap_distribution(by_config: Dict, output_file: str):
    """Box plot of gap distribution by configuration."""
    # Extract gap data
    data = []
    labels = []
    
    for config_id in sorted(by_config.keys()):
        rows = by_config[config_id]
        gaps = []
        for r in rows:
            if r["gap_to_optimal_percent"] not in ["N/A", None, ""]:
                try:
                    gap = float(r["gap_to_optimal_percent"])
                    if gap > 0:  # Only include non-optimal cases
                        gaps.append(gap)
                except:
                    pass
        
        if gaps:
            data.append(gaps)
            labels.append(config_id)
    
    if not data:
        print("No gap data to plot")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Configuration (Tasks_QCs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap to Optimal (%)', fontsize=12, fontweight='bold')
    ax.set_title('Heuristic Gap Distribution (Non-Optimal Cases)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_runtime_comparison(by_config: Dict, output_file: str):
    """Scatter plot comparing MILP vs heuristic runtime."""
    milp_times = []
    heur_times = []
    
    for config_id in sorted(by_config.keys()):
        rows = by_config[config_id]
        for r in rows:
            if r["milp_runtime"] != "N/A" and r["heuristic_runtime"] != "N/A":
                try:
                    milp_times.append(float(r["milp_runtime"]))
                    heur_times.append(float(r["heuristic_runtime"]))
                except:
                    pass
    
    if not milp_times:
        print("No runtime data to plot")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(milp_times, heur_times, alpha=0.6, s=50)
    
    # Add diagonal line
    max_val = max(max(milp_times), max(heur_times))
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal runtime')
    
    ax.set_xlabel('MILP Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Heuristic Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Runtime Comparison: MILP vs Heuristic', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_part_b(input_dir: str, figures_dir: str):
    """Generate plots for Part B results."""
    input_path = Path(input_dir)
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    runs_file = input_path / "partB_runs.csv"
    timeseries_file = input_path / "partB_timeseries.csv"
    
    if not runs_file.exists():
        print(f"Part B runs not found: {runs_file}")
        return
    
    runs = load_csv(str(runs_file))
    
    # Plot 1: Winner distribution
    plot_winner_distribution(runs, str(figures_path / "partB_winner_distribution.png"))
    
    # Plot 2: Time series for representative instances
    if timeseries_file.exists():
        timeseries = load_csv(str(timeseries_file))
        plot_representative_timeseries(timeseries, runs, str(figures_path / "partB_timeseries"), max_plots=None)
    
    # Plot 3: Checkpoint comparison
    plot_checkpoint_comparison(runs, str(figures_path / "partB_checkpoint_comparison.png"))
    
    print(f"Part B plots saved to {figures_path}")


def plot_winner_distribution(runs: List[Dict], output_file: str):
    """Bar plot of winner distribution by configuration."""
    from collections import Counter
    
    by_config = defaultdict(list)
    for row in runs:
        config_id = row["config_id"]
        by_config[config_id].append(row)
    
    # Count winners per config
    config_winners = {}
    for config_id, rows in sorted(by_config.items()):
        counter = Counter(r["winner"] for r in rows)
        config_winners[config_id] = counter
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    configs = list(config_winners.keys())
    milp_counts = [config_winners[c].get("MILP", 0) for c in configs]
    heur_counts = [config_winners[c].get("HEURISTIC", 0) for c in configs]
    tie_counts = [config_winners[c].get("TIE", 0) for c in configs]
    
    x = np.arange(len(configs))
    width = 0.25
    
    ax.bar(x - width, milp_counts, width, label='MILP', alpha=0.8, color='red')
    ax.bar(x, heur_counts, width, label='Heuristic', alpha=0.8, color='blue')
    ax.bar(x + width, tie_counts, width, label='Tie', alpha=0.8, color='green')
    
    ax.set_xlabel('Configuration (Tasks_QCs)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Wins', fontsize=12, fontweight='bold')
    ax.set_title('Solution Quality: Winner Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_representative_timeseries(timeseries: List[Dict], runs: List[Dict], output_prefix: str, max_plots: int = None):
    """Plot time series for representative instances (all if max_plots is None)."""
    # Group timeseries by instance
    by_instance = defaultdict(lambda: {"MILP": [], "HEURISTIC": []})
    
    for entry in timeseries:
        inst_id = entry["instance_id"]
        method = entry["method"]
        by_instance[inst_id][method].append(entry)
    
    # Select representative instances (different configs, mix of winners)
    selected_instances = []
    configs_seen = set()
    
    for run in runs:
        inst_id = run["instance_id"]
        config_id = run["config_id"]
        
        limit_reached = (max_plots is not None and len(selected_instances) >= max_plots)
        if config_id not in configs_seen and not limit_reached:
            selected_instances.append((inst_id, run))
            configs_seen.add(config_id)
    
    # Plot each instance
    for inst_id, run in selected_instances:
        plot_single_timeseries(
            by_instance[inst_id],
            run,
            f"{output_prefix}_inst{inst_id}.png"
        )


def plot_single_timeseries(instance_data: Dict, run: Dict, output_file: str):
    """Plot time series for a single instance."""
    fig, ax = plt.subplots(figsize=(12, 7))
    plotted_times = []
    
    # Plot MILP
    milp_entries = sorted(instance_data["MILP"], key=lambda x: float(x.get("time", 0)))
    if milp_entries:
        times = []
        objs = []
        for e in milp_entries:
            if e.get("objective") in ["N/A", None, ""]:
                continue
            try:
                times.append(float(e["time"]))
                objs.append(float(e["objective"]))
            except (ValueError, TypeError, KeyError):
                continue
        if times and objs:
            # Create step plot (best-so-far)
            best_times = []
            best_objs = []
            current_best = float('inf')
            
            for t, obj in zip(times, objs):
                if obj < current_best:
                    current_best = obj
                    best_times.append(t)
                    best_objs.append(obj)
            
            if best_objs:
                ax.step(best_times, best_objs, where='post', label='MILP', 
                       color='red', linewidth=2.5, alpha=0.8)
                plotted_times.extend(best_times)
    
    # Plot Heuristic
    heur_entries = sorted(instance_data["HEURISTIC"], key=lambda x: float(x.get("time", 0)))
    if heur_entries:
        times = []
        objs = []
        for e in heur_entries:
            if e.get("objective") in ["N/A", None, ""]:
                continue
            try:
                times.append(float(e["time"]))
                objs.append(float(e["objective"]))
            except (ValueError, TypeError, KeyError):
                continue
        if times and objs:
            ax.step(times, objs, where='post', label='Heuristic', 
                   color='blue', linewidth=2.5, alpha=0.75)
            plotted_times.extend(times)

    # Long-tail runtimes can flatten the early race; use symlog when span is large.
    positive_times = [t for t in plotted_times if t > 0]
    if positive_times:
        min_positive = min(positive_times)
        max_time = max(positive_times)

        if max_time / max(min_positive, 1e-6) >= 20:
            linthresh = max(1.0, min_positive)
            ax.set_xscale('symlog', linthresh=linthresh, linscale=1.0)

        ax.set_xlim(left=0, right=max_time * 1.05)
    
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Objective Value', fontsize=12, fontweight='bold')
    ax.set_title(f"Instance {run['instance_id']}: {run['config_id']} "
                f"(Winner: {run['winner']})", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_checkpoint_comparison(runs: List[Dict], output_file: str):
    """Plot objective values at time checkpoints."""
    # Extract checkpoint columns
    checkpoint_cols = [c for c in runs[0].keys() if c.startswith("milp_obj_at_") or c.startswith("heur_obj_at_")]
    
    if not checkpoint_cols:
        print("No checkpoint data found")
        return
    
    # Extract checkpoint times
    checkpoints = sorted(set(int(c.split("_at_")[1].replace("s", "")) 
                            for c in checkpoint_cols if "_at_" in c))
    
    # Calculate average objectives at each checkpoint
    milp_avgs = []
    heur_avgs = []
    
    for cp in checkpoints:
        milp_col = f"milp_obj_at_{cp}s"
        heur_col = f"heur_obj_at_{cp}s"
        
        milp_vals = [float(r[milp_col]) for r in runs if r.get(milp_col) not in ["N/A", None, ""]]
        heur_vals = [float(r[heur_col]) for r in runs if r.get(heur_col) not in ["N/A", None, ""]]
        
        milp_avgs.append(np.mean(milp_vals) if milp_vals else None)
        heur_avgs.append(np.mean(heur_vals) if heur_vals else None)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(checkpoints, milp_avgs, 'o-', label='MILP', color='red', 
           linewidth=2.5, markersize=8, alpha=0.8)
    ax.plot(checkpoints, heur_avgs, 's-', label='Heuristic', color='blue', 
           linewidth=2.5, markersize=8, alpha=0.75)
    
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Objective Value', fontsize=12, fontweight='bold')
    ax.set_title('Average Performance at Time Checkpoints', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
