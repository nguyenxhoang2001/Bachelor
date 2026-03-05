"""
Part A: Optimality Check Suite (small instances).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from typing import Dict, List
import numpy as np

from benchmark.instance_factory import InstanceFactory
from benchmark.common import Timer, CSVWriter, format_value, load_yaml, ensure_pkg_path
from benchmark.milp_logger import solve_milp_with_tracking
from benchmark.heuristic_logger import solve_heuristic_with_tracking

ensure_pkg_path()


def run_part_a(config_path: str, output_dir: str):
    """
    Run Part A benchmark: optimality checking on small instances.
    """
    # Load config
    config = load_yaml(config_path)
    
    part_a_config = config["part_a"]
    instance_config = config["instance"]
    milp_config = config["milp"]
    heuristic_config = config["heuristic"]
    
    print("="*70)
    print("PART A: OPTIMALITY CHECK SUITE")
    print("="*70)
    
    # Create instance factory
    factory = InstanceFactory(instance_config)
    
    # Generate instances
    print(f"\nGenerating instances...")
    instances = factory.generate_suite(
        task_sizes=part_a_config["task_sizes"],
        qc_counts=part_a_config["qc_counts"],
        num_instances_per_config=part_a_config["num_instances_per_config"],
        seed_base=part_a_config.get("seed_base", 42)
    )
    print(f"Generated {len(instances)} instances")
    
    # Prepare CSV writers
    output_path = Path(output_dir)
    results_csv = CSVWriter(
        str(output_path / "partA_results.csv"),
        fieldnames=[
            "instance_id", "config_id", "num_tasks", "num_qcs", "num_bays", "seed",
            "milp_status", "milp_objective", "milp_best_bound", "milp_gap",
            "milp_runtime", "milp_node_count", "milp_sol_count",
            "heuristic_objective", "heuristic_runtime", "heuristic_iterations",
            "heuristic_feasible", "is_optimal_match", "gap_to_optimal_percent"
        ]
    )
    
    # Run benchmark
    print(f"\nRunning benchmark...")
    eps = 1e-6  # Tolerance for optimality match
    
    for i, (problem, metadata) in enumerate(instances):
        print(f"\n[{i+1}/{len(instances)}] Instance {metadata['instance_id']}: "
              f"{metadata['num_tasks']} tasks, {metadata['num_qcs']} QCs")
        
        # Run MILP
        print(f"  Running MILP...")
        timer = Timer().start()
        milp_result, milp_trace = solve_milp_with_tracking(
            problem,
            time_limit=milp_config["time_limit"],
            threads=milp_config["threads"]
        )
        milp_result["runtime"] = timer.stop()
        
        print(f"  MILP: status={milp_result['status']}, "
              f"obj={milp_result['objective']}, "
              f"runtime={milp_result['runtime']:.2f}s")
        
        # Run Heuristic
        print(f"  Running Heuristic...")
        timer = Timer().start()
        heur_result, heur_trace = solve_heuristic_with_tracking(
            problem,
            time_limit=heuristic_config["time_limit"],
            max_iterations=heuristic_config["max_iterations"],
            no_improve_limit=heuristic_config.get("no_improve_limit"),
            seed=heuristic_config.get("seed", 42)
        )
        heur_result["runtime"] = timer.stop()
        
        print(f"  Heuristic: obj={heur_result['objective']}, "
              f"runtime={heur_result['runtime']:.2f}s, "
              f"feasible={heur_result['is_feasible']}")
        
        # Check optimality match
        is_optimal_match = False
        gap_to_optimal = None
        
        if (milp_result["status"] == "OPTIMAL" and 
            milp_result["objective"] is not None and 
            heur_result["objective"] is not None and
            heur_result["is_feasible"]):
            
            rel_diff = abs(milp_result["objective"] - heur_result["objective"]) / \
                      max(abs(milp_result["objective"]), eps)
            
            is_optimal_match = rel_diff < eps
            # Report unsigned deviation from optimal to avoid negative gaps due to
            # tiny numerical differences between solvers.
            gap_to_optimal = abs(heur_result["objective"] - milp_result["objective"]) / \
                           max(abs(milp_result["objective"]), eps) * 100
        
        # Record results
        results_csv.add_row({
            "instance_id": metadata["instance_id"],
            "config_id": metadata["config_id"],
            "num_tasks": metadata["num_tasks"],
            "num_qcs": metadata["num_qcs"],
            "num_bays": metadata["num_bays"],
            "seed": metadata["seed"],
            "milp_status": milp_result["status"],
            "milp_objective": format_value(milp_result["objective"]),
            "milp_best_bound": format_value(milp_result["best_bound"]),
            "milp_gap": format_value(milp_result["gap"]),
            "milp_runtime": format_value(milp_result["runtime"]),
            "milp_node_count": milp_result["node_count"],
            "milp_sol_count": milp_result["sol_count"],
            "heuristic_objective": format_value(heur_result["objective"]),
            "heuristic_runtime": format_value(heur_result["runtime"]),
            "heuristic_iterations": heur_result["iterations"],
            "heuristic_feasible": heur_result["is_feasible"],
            "is_optimal_match": is_optimal_match,
            "gap_to_optimal_percent": format_value(gap_to_optimal),
        })
    
    # Write results
    results_csv.flush()
    
    # Generate summary
    print(f"\nGenerating summary...")
    generate_part_a_summary(results_csv.rows, str(output_path / "partA_summary.csv"))
    
    print(f"\n{'='*70}")
    print(f"PART A COMPLETE")
    print(f"Results: {output_path / 'partA_results.csv'}")
    print(f"Summary: {output_path / 'partA_summary.csv'}")
    print(f"{'='*70}\n")


def generate_part_a_summary(rows: List[Dict], output_file: str):
    """Generate aggregate summary by (T, K) configuration."""
    from collections import defaultdict
    
    # Group by config_id
    by_config = defaultdict(list)
    for row in rows:
        by_config[row["config_id"]].append(row)
    
    summary_rows = []
    
    for config_id, config_rows in sorted(by_config.items()):
        # Parse config
        num_tasks = config_rows[0]["num_tasks"]
        num_qcs = config_rows[0]["num_qcs"]
        n = len(config_rows)
        
        # Calculate metrics
        optimal_matches = sum(1 for r in config_rows if r["is_optimal_match"])
        match_rate = optimal_matches / n * 100 if n > 0 else 0
        
        # Gap statistics (only for feasible, non-matching)
        gaps = []
        for r in config_rows:
            if (r["gap_to_optimal_percent"] not in ["N/A", None] and 
                not r["is_optimal_match"]):
                try:
                    gaps.append(float(r["gap_to_optimal_percent"]))
                except:
                    pass
        
        mean_gap = np.mean(gaps) if gaps else None
        std_gap = np.std(gaps) if gaps else None
        
        # Runtime statistics
        milp_runtimes = [float(r["milp_runtime"]) for r in config_rows if r["milp_runtime"] != "N/A"]
        heur_runtimes = [float(r["heuristic_runtime"]) for r in config_rows if r["heuristic_runtime"] != "N/A"]
        
        summary_rows.append({
            "config_id": config_id,
            "num_tasks": num_tasks,
            "num_qcs": num_qcs,
            "num_instances": n,
            "optimal_match_count": optimal_matches,
            "optimal_match_rate_percent": f"{match_rate:.1f}",
            "mean_gap_to_optimal_percent": format_value(mean_gap, 2),
            "std_gap_to_optimal_percent": format_value(std_gap, 2),
            "mean_milp_runtime": format_value(np.mean(milp_runtimes), 2) if milp_runtimes else "N/A",
            "mean_heuristic_runtime": format_value(np.mean(heur_runtimes), 2) if heur_runtimes else "N/A",
        })
    
    # Write summary
    writer = CSVWriter(output_file, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
    writer.rows = summary_rows
    writer.flush()
