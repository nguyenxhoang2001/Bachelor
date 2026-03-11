import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from typing import Dict, List
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from benchmark.instance_factory import InstanceFactory
from benchmark.common import Timer, CSVWriter, format_value, load_yaml, ensure_pkg_path
from benchmark.milp_logger import solve_milp_with_tracking
from benchmark.heuristic_logger import solve_heuristic_with_tracking

ensure_pkg_path()


def build_part_b_fieldnames(checkpoints: List[float]) -> List[str]:
    fieldnames = [
        "instance_id", "config_id", "num_tasks", "num_qcs", "num_bays", "seed",
        "milp_status", "milp_final_obj", "milp_best_bound", "milp_gap",
        "milp_runtime", "milp_node_count", "milp_sol_count",
        "heuristic_final_obj", "heuristic_runtime", "heuristic_iterations",
        "heuristic_feasible", "winner", "gap_milp_heur_percent"
    ]

    for cp in checkpoints:
        fieldnames.extend([f"milp_obj_at_{cp}s", f"heur_obj_at_{cp}s"])

    return fieldnames


def run_part_b_single_config(
    num_tasks: int,
    num_qcs: int,
    num_instances_per_config: int,
    seed_base: int,
    global_start_instance_id: int,
    checkpoints: List[float],
    instance_config: Dict,
    milp_config: Dict,
    heuristic_config: Dict,
) -> Dict:
    factory = InstanceFactory(instance_config)
    run_rows = []
    timeseries_rows = []

    for i in range(num_instances_per_config):
        instance_id = global_start_instance_id + i
        seed = seed_base + instance_id

        problem, metadata = factory.generate(num_tasks, num_qcs, seed)
        metadata["instance_id"] = instance_id
        metadata["config_id"] = f"T{num_tasks}_K{num_qcs}"

        # Run MILP
        timer = Timer().start()
        milp_result, milp_trace = solve_milp_with_tracking(
            problem,
            time_limit=milp_config["time_limit"],
            threads=milp_config["threads"]
        )
        milp_result["runtime"] = timer.stop()

        # Run Heuristic
        timer = Timer().start()
        heur_result, heur_trace = solve_heuristic_with_tracking(
            problem,
            time_limit=heuristic_config["time_limit"],
            max_iterations=heuristic_config["max_iterations"],
            no_improve_limit=heuristic_config.get("no_improve_limit"),
            seed=heuristic_config.get("seed", 42)
        )
        heur_result["runtime"] = timer.stop()

        milp_checkpoints = extract_checkpoints(milp_trace, checkpoints, "incumbent")
        heur_checkpoints = extract_checkpoints(heur_trace, checkpoints, "best_objective")

        winner = determine_winner(
            milp_result["objective"],
            heur_result["objective"] if heur_result["is_feasible"] else None
        )

        gap_milp_heur = None
        if (milp_result["objective"] is not None and
            heur_result["objective"] is not None and
            heur_result["is_feasible"]):
            gap_milp_heur = (heur_result["objective"] - milp_result["objective"]) / \
                           max(abs(heur_result["objective"]), 1e-6) * 100

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

        for cp in checkpoints:
            result_row[f"milp_obj_at_{cp}s"] = format_value(milp_checkpoints.get(cp))
            result_row[f"heur_obj_at_{cp}s"] = format_value(heur_checkpoints.get(cp))

        run_rows.append(result_row)

        for entry in milp_trace:
            timeseries_rows.append({
                "instance_id": instance_id,
                "method": "MILP",
                "time": format_value(entry.get("time", 0.0)),
                "objective": format_value(entry.get("incumbent")),
                "best_bound": format_value(entry.get("best_bound")),
                "gap": format_value(entry.get("gap")),
            })

        for entry in heur_trace:
            timeseries_rows.append({
                "instance_id": instance_id,
                "method": "HEURISTIC",
                "time": format_value(entry.get("time", 0.0)),
                "objective": format_value(entry.get("best_objective")),
                "best_bound": "N/A",
                "gap": "N/A",
            })

    return {
        "config_id": f"T{num_tasks}_K{num_qcs}",
        "run_rows": run_rows,
        "timeseries_rows": timeseries_rows,
    }


def run_part_b(config_path: str, output_dir: str):
    # Load config
    config = load_yaml(config_path)
    
    part_b_config = config["part_b"]
    instance_config = config["instance"]
    milp_config = config["milp"]
    heuristic_config = config["heuristic"]
    checkpoints = part_b_config.get("checkpoints", [1, 5, 10, 30, 60, 120, 300, 600])
    parallel_workers = max(1, int(part_b_config.get("parallel_workers", 1)))
    
    print("="*70)
    print("PART B: PRIMAL BOUND RACE SUITE")
    print("="*70)
    
    config_pairs = [
        (num_tasks, num_qcs)
        for num_tasks in part_b_config["task_sizes"]
        for num_qcs in part_b_config["qc_counts"]
    ]
    total_instances = len(config_pairs) * part_b_config["num_instances_per_config"]

    print(f"\nPreparing {len(config_pairs)} Part B configs with {total_instances} total instances")
    print(f"Parallel workers: {parallel_workers}")
    if parallel_workers > 1 and milp_config.get("threads", 1) > 1:
        print("Warning: milp.threads > 1 with parallel workers may oversubscribe CPU")
    
    # Prepare CSV writers
    output_path = Path(output_dir)
    
    # Main results CSV
    results_csv = CSVWriter(
        str(output_path / "partB_runs.csv"),
        build_part_b_fieldnames(checkpoints)
    )
    
    # Time series CSV (long format)
    timeseries_csv = CSVWriter(
        str(output_path / "partB_timeseries.csv"),
        fieldnames=["instance_id", "method", "time", "objective", "best_bound", "gap"]
    )
    
    print(f"\nRunning benchmark...")

    num_instances_per_config = part_b_config["num_instances_per_config"]
    seed_base = part_b_config.get("seed_base", 1000)

    if parallel_workers == 1:
        for cfg_idx, (num_tasks, num_qcs) in enumerate(config_pairs):
            start_id = cfg_idx * num_instances_per_config
            print(
                f"  Running config T{num_tasks}_K{num_qcs} "
                f"({cfg_idx + 1}/{len(config_pairs)})"
            )
            payload = run_part_b_single_config(
                num_tasks=num_tasks,
                num_qcs=num_qcs,
                num_instances_per_config=num_instances_per_config,
                seed_base=seed_base,
                global_start_instance_id=start_id,
                checkpoints=checkpoints,
                instance_config=instance_config,
                milp_config=milp_config,
                heuristic_config=heuristic_config,
            )
            payload["run_rows"].sort(key=lambda row: int(row["instance_id"]))
            payload["timeseries_rows"].sort(
                key=lambda row: (
                    int(row["instance_id"]),
                    row["method"],
                    float(row["time"]) if row["time"] != "N/A" else 0.0,
                )
            )
            for row in payload["run_rows"]:
                results_csv.add_row(row)
            for row in payload["timeseries_rows"]:
                timeseries_csv.add_row(row)
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_cfg = {}
            for cfg_idx, (num_tasks, num_qcs) in enumerate(config_pairs):
                start_id = cfg_idx * num_instances_per_config
                future = executor.submit(
                    run_part_b_single_config,
                    num_tasks,
                    num_qcs,
                    num_instances_per_config,
                    seed_base,
                    start_id,
                    checkpoints,
                    instance_config,
                    milp_config,
                    heuristic_config,
                )
                future_to_cfg[future] = (num_tasks, num_qcs)

            completed = 0
            for future in as_completed(future_to_cfg):
                num_tasks, num_qcs = future_to_cfg[future]
                payload = future.result()
                payload["run_rows"].sort(key=lambda row: int(row["instance_id"]))
                payload["timeseries_rows"].sort(
                    key=lambda row: (
                        int(row["instance_id"]),
                        row["method"],
                        float(row["time"]) if row["time"] != "N/A" else 0.0,
                    )
                )
                for row in payload["run_rows"]:
                    results_csv.add_row(row)
                for row in payload["timeseries_rows"]:
                    timeseries_csv.add_row(row)
                completed += 1
                print(
                    f"  Finished config T{num_tasks}_K{num_qcs} "
                    f"({completed}/{len(config_pairs)})"
                )
    
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
        for row in summary_rows:
            writer.add_row(row)
        writer.flush()
