"""
Part B: Primal Bound Race Suite (larger instances with time-series analysis).
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


def run_part_b(config_path: str, output_dir: str):
    """
    Run Part B benchmark: primal bound race on larger instances.
    """
    # Load config
    config = load_yaml(config_path)
    
    part_b_config = config["part_b"]
    instance_config = config["instance"]
    milp_config = config["milp"]
    heuristic_config = config["heuristic"]
    checkpoints = part_b_config.get("checkpoints", [1, 5, 10, 30, 60, 120, 300, 600])
    
    print("="*70)
    print("PART B: PRIMAL BOUND RACE SUITE")
    print("="*70)
    
    # Create instance factory
    factory = InstanceFactory(instance_config)
    
    # Generate instances
    print(f"\nGenerating instances...")
    instances = factory.generate_suite(
        task_sizes=part_b_config["task_sizes"],
        qc_counts=part_b_config["qc_counts"],
        num_instances_per_config=part_b_config["num_instances_per_config"],
        seed_base=part_b_config.get("seed_base", 1000)
    )
    print(f"Generated {len(instances)} instances")
    
    # Prepare CSV writers
    output_path = Path(output_dir)
    
    # Main results CSV
    results_fieldnames = [
        "instance_id", "config_id", "num_tasks", "num_qcs", "num_bays", "seed",
        "milp_status", "milp_final_obj", "milp_best_bound", "milp_gap",
        "milp_runtime", "milp_node_count", "milp_sol_count",
        "heuristic_final_obj", "heuristic_runtime", "heuristic_iterations",
        "heuristic_feasible", "winner", "gap_milp_heur_percent"
    ]
    
    # Add checkpoint columns
    for cp in checkpoints:
        results_fieldnames.extend([f"milp_obj_at_{cp}s", f"heur_obj_at_{cp}s"])
    
    results_csv = CSVWriter(str(output_path / "partB_runs.csv"), results_fieldnames)
    
    # Time series CSV (long format)
    timeseries_csv = CSVWriter(
        str(output_path / "partB_timeseries.csv"),
        fieldnames=["instance_id", "method", "time", "objective", "best_bound", "gap"]
    )
    
    # Run benchmark
    print(f"\nRunning benchmark...")
    
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
        
        # Extract objectives at checkpoints
        milp_checkpoints = extract_checkpoints(milp_trace, checkpoints, "incumbent")
        heur_checkpoints = extract_checkpoints(heur_trace, checkpoints, "best_objective")
        
        # Determine winner
        winner = determine_winner(
            milp_result["objective"],
            heur_result["objective"] if heur_result["is_feasible"] else None
        )
        
        # Calculate gap
        gap_milp_heur = None
        if (milp_result["objective"] is not None and 
            heur_result["objective"] is not None and 
            heur_result["is_feasible"]):
            gap_milp_heur = (heur_result["objective"] - milp_result["objective"]) / \
                           max(abs(heur_result["objective"]), 1e-6) * 100
        
        # Build result row
        result_row = {
            "instance_id": metadata["instance_id"],
            "config_id": metadata["config_id"],
            "num_tasks": metadata["num_tasks"],
            "num_qcs": metadata["num_qcs"],
            "num_bays": metadata["num_bays"],
            "seed": metadata["seed"],
            "milp_status": milp_result["status"],
            "milp_final_obj": format_value(milp_result["objective"]),
            "milp_best_bound": format_value(milp_result["best_bound"]),
            "milp_gap": format_value(milp_result["gap"]),
            "milp_runtime": format_value(milp_result["runtime"]),
            "milp_node_count": milp_result["node_count"],
            "milp_sol_count": milp_result["sol_count"],
            "heuristic_final_obj": format_value(heur_result["objective"]),
            "heuristic_runtime": format_value(heur_result["runtime"]),
            "heuristic_iterations": heur_result["iterations"],
            "heuristic_feasible": heur_result["is_feasible"],
            "winner": winner,
            "gap_milp_heur_percent": format_value(gap_milp_heur),
        }
        
        # Add checkpoint values
        for cp in checkpoints:
            result_row[f"milp_obj_at_{cp}s"] = format_value(milp_checkpoints.get(cp))
            result_row[f"heur_obj_at_{cp}s"] = format_value(heur_checkpoints.get(cp))
        
        results_csv.add_row(result_row)
        
        # Log time series
        instance_id = metadata["instance_id"]
        
        for entry in milp_trace:
            timeseries_csv.add_row({
                "instance_id": instance_id,
                "method": "MILP",
                "time": format_value(entry.get("time", 0.0)),
                "objective": format_value(entry.get("incumbent")),
                "best_bound": format_value(entry.get("best_bound")),
                "gap": format_value(entry.get("gap")),
            })
        
        for entry in heur_trace:
            timeseries_csv.add_row({
                "instance_id": instance_id,
                "method": "HEURISTIC",
                "time": format_value(entry.get("time", 0.0)),
                "objective": format_value(entry.get("best_objective")),
                "best_bound": "N/A",
                "gap": "N/A",
            })
    
    # Write results
    results_csv.flush()
    timeseries_csv.flush()
    
    # Generate summary
    print(f"\nGenerating summary...")
    generate_part_b_summary(results_csv.rows, str(output_path / "partB_summary.csv"), checkpoints)
    
    print(f"\n{'='*70}")
    print(f"PART B COMPLETE")
    print(f"Results: {output_path / 'partB_runs.csv'}")
    print(f"Time series: {output_path / 'partB_timeseries.csv'}")
    print(f"Summary: {output_path / 'partB_summary.csv'}")
    print(f"{'='*70}\n")


def extract_checkpoints(trace: List[Dict], checkpoints: List[float], key: str) -> Dict[float, float]:
    """
    Extract objective values at specific time checkpoints from trace.
    
    Returns dict mapping checkpoint -> objective value
    """
    checkpoint_values = {}
    
    if not trace:
        return checkpoint_values
    
    # Sort trace by time
    sorted_trace = sorted(trace, key=lambda x: x.get("time", 0.0))
    
    for cp in checkpoints:
        # Find best objective up to checkpoint time
        best_obj = None
        for entry in sorted_trace:
            if entry.get("time", 0.0) <= cp:
                obj = entry.get(key)
                if obj is not None:
                    if best_obj is None or obj < best_obj:
                        best_obj = obj
            else:
                break
        
        checkpoint_values[cp] = best_obj
    
    return checkpoint_values


def determine_winner(milp_obj, heur_obj) -> str:
    """Determine which method achieved better objective."""
    if milp_obj is None and heur_obj is None:
        return "NONE"
    elif milp_obj is None:
        return "HEURISTIC"
    elif heur_obj is None:
        return "MILP"
    elif abs(milp_obj - heur_obj) < 1e-6:
        return "TIE"
    elif milp_obj < heur_obj:
        return "MILP"
    else:
        return "HEURISTIC"


def generate_part_b_summary(rows: List[Dict], output_file: str, checkpoints: List[float]):
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
        
        # Count winners
        winner_counts = {"MILP": 0, "HEURISTIC": 0, "TIE": 0, "NONE": 0}
        for r in config_rows:
            winner_counts[r["winner"]] += 1
        
        # Runtime statistics
        milp_runtimes = [float(r["milp_runtime"]) for r in config_rows if r["milp_runtime"] != "N/A"]
        heur_runtimes = [float(r["heuristic_runtime"]) for r in config_rows if r["heuristic_runtime"] != "N/A"]
        
        # Gap statistics
        gaps = []
        for r in config_rows:
            if r["gap_milp_heur_percent"] not in ["N/A", None]:
                try:
                    gaps.append(float(r["gap_milp_heur_percent"]))
                except:
                    pass
        
        summary_row = {
            "config_id": config_id,
            "num_tasks": num_tasks,
            "num_qcs": num_qcs,
            "num_instances": n,
            "milp_wins": winner_counts["MILP"],
            "heuristic_wins": winner_counts["HEURISTIC"],
            "ties": winner_counts["TIE"],
            "mean_milp_runtime": format_value(np.mean(milp_runtimes), 2) if milp_runtimes else "N/A",
            "mean_heuristic_runtime": format_value(np.mean(heur_runtimes), 2) if heur_runtimes else "N/A",
            "mean_gap_percent": format_value(np.mean(gaps), 2) if gaps else "N/A",
            "median_gap_percent": format_value(np.median(gaps), 2) if gaps else "N/A",
        }
        
        summary_rows.append(summary_row)
    
    # Write summary
    if summary_rows:
        writer = CSVWriter(output_file, fieldnames=list(summary_rows[0].keys()))
        writer.rows = summary_rows
        writer.flush()
