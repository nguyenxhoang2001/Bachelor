from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from qc_problem import QCProblem
from milp_solver import solve_milp
from tabu_solver import solve_tabu
from greedysolver import check_interference

DEFAULT_CONFIG: Dict[str, object] = {
    "instance": {
        "processing_time_range": [3.0, 180.0],
        "travel_per_bay": 1.0,
        "precedence_probability": 0.3,
        "num_bays_mode": "proportional",
        "fixed_bays": 30,
    },
    "milp": {
        "time_limit": 1800.0,
        "threads": 1,
    },
    "heuristic": {
        "time_limit": 1800.0,
        "max_iterations": 200000,
        "no_improve_limit": 5000,
        "seed": 42,
    },
    "part_a": {
        "task_sizes": [4, 5, 6, 7, 8],
        "qc_counts": [2, 3],
        "num_instances_per_config": 5,
        "seed_base": 42,
        "time_limit": 1000.0,
    },
    "part_b": {
        "task_sizes": [20, 25, 30, 35, 40],
        "qc_counts": [2, 3],
        "num_instances_per_config": 10,
        "seed_base": 1000,
        "time_limit": 1800.0,
        "checkpoints": [1, 5, 10, 30, 60, 120, 300, 600, 1000, 1400, 1800],
    },
}

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def format_value(value, precision: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)

def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written {len(rows)} rows to {path}")

def num_bays_for_tasks(instance_cfg: Dict[str, object], num_tasks: int) -> int:
    if instance_cfg.get("num_bays_mode") == "proportional":
        return int(num_tasks)
    return int(instance_cfg.get("fixed_bays", 30))

def generate_suite(
    instance_cfg: Dict[str, object],
    task_sizes: Sequence[int],
    qc_counts: Sequence[int],
    num_instances_per_config: int,
    seed_base: int,
) -> List[Tuple[QCProblem, Dict[str, object]]]:
    instances: List[Tuple[QCProblem, Dict[str, object]]] = []
    instance_id = 0
    for num_tasks in task_sizes:
        for num_qcs in qc_counts:
            for _ in range(num_instances_per_config):
                seed = seed_base + instance_id
                num_bays = num_bays_for_tasks(instance_cfg, num_tasks)
                problem = QCProblem.generate(
                    num_bays=num_bays,
                    num_qcs=num_qcs,
                    total_tasks=num_tasks,
                    processing_time_range=tuple(instance_cfg["processing_time_range"]),
                    travel_per_bay=float(instance_cfg["travel_per_bay"]),
                    precedence_probability=float(instance_cfg["precedence_probability"]),
                    seed=seed,
                )
                metadata = {
                    "instance_id": instance_id,
                    "config_id": f"T{num_tasks}_K{num_qcs}",
                    "num_tasks": num_tasks,
                    "num_qcs": num_qcs,
                    "num_bays": num_bays,
                    "seed": seed,
                }
                instances.append((problem, metadata))
                instance_id += 1
    return instances

def parse_gurobi_log_timeseries(log_file: str) -> List[Dict[str, object]]:
    timeseries: List[Dict[str, object]] = []
    if not os.path.exists(log_file):
        return timeseries

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        if "Found heuristic solution: objective" in line:
            match = re.search(r"objective\s+([\d.eE+\-]+)", line)
            if match:
                timeseries.append(
                    {
                        "time": 0.0,
                        "incumbent": float(match.group(1)),
                        "best_bound": None,
                        "gap": None,
                    }
                )

        stripped = line.strip()
        if not stripped.startswith("H") and not stripped.startswith("*"):
            continue

        parts = stripped.split()
        if len(parts) < 7:
            continue

        try:
            t = float(parts[-1].replace("s", ""))
        except ValueError:
            continue

        for i, part in enumerate(parts):
            if "%" not in part or i < 2:
                continue
            try:
                incumbent = float(parts[i - 2])
                best_bound = float(parts[i - 1])
                gap = float(part.replace("%", "")) if part.replace("%", "") else None
                timeseries.append(
                    {
                        "time": t,
                        "incumbent": incumbent,
                        "best_bound": best_bound,
                        "gap": gap,
                    }
                )
                break
            except ValueError:
                continue

    return timeseries

def solve_milp_with_tracking(problem: QCProblem, time_limit: float) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    fd, log_file = tempfile.mkstemp(suffix=".log", text=True)
    os.close(fd)
    try:
        start = time.time()
        result = solve_milp(problem, time_limit=time_limit, verbose=False, log_file=log_file)
        runtime = time.time() - start
        trace = parse_gurobi_log_timeseries(log_file)
        out = {
            "status": result.get("status", "UNKNOWN"),
            "objective": result.get("objective"),
            "best_bound": result.get("best_bound"),
            "gap": result.get("gap"),
            "runtime": runtime,
            "node_count": result.get("node_count", 0),
            "sol_count": result.get("sol_count", 0),
        }
        return out, trace
    finally:
        try:
            os.remove(log_file)
        except OSError:
            pass

def solve_heuristic_with_tracking(
    problem: QCProblem,
    time_limit: float,
    max_iterations: int,
    no_improve_limit: Optional[int],
    seed: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    start = time.time()
    schedule, objective, trace = solve_tabu(
        problem,
        max_iteration=max_iterations,
        max_time=time_limit,
        no_improve_limit=no_improve_limit,
        seed=seed,
        verbose=False,
        return_trace=True,
    )
    runtime = time.time() - start
    feasible, violation = check_interference(problem, schedule)

    timeseries = [{"time": e.get("time", 0.0), "best_objective": e.get("best_objective", objective)} for e in trace]
    result = {
        "objective": objective if feasible else None,
        "runtime": runtime,
        "iterations": len(trace),
        "is_feasible": feasible,
        "violation": None if feasible else violation,
    }
    return result, timeseries

def extract_checkpoints(trace: List[Dict[str, object]], checkpoints: Sequence[float], key: str) -> Dict[float, Optional[float]]:
    cp_values: Dict[float, Optional[float]] = {}
    if not trace:
        for cp in checkpoints:
            cp_values[cp] = None
        return cp_values

    sorted_trace = sorted(trace, key=lambda x: float(x.get("time", 0.0)))
    for cp in checkpoints:
        best_obj = None
        for entry in sorted_trace:
            t = float(entry.get("time", 0.0))
            if t > cp:
                break
            obj = entry.get(key)
            if obj is not None and (best_obj is None or float(obj) < float(best_obj)):
                best_obj = float(obj)
        cp_values[cp] = best_obj
    return cp_values

def determine_winner(milp_obj: Optional[float], heur_obj: Optional[float]) -> str:
    if milp_obj is None and heur_obj is None:
        return "NONE"
    if milp_obj is None:
        return "HEURISTIC"
    if heur_obj is None:
        return "MILP"
    if abs(milp_obj - heur_obj) < 1e-6:
        return "TIE"
    return "MILP" if milp_obj < heur_obj else "HEURISTIC"

def run_part_a(config: Dict[str, object], output_dir: str) -> None:
    print("=" * 70)
    print("PART A: OPTIMALITY CHECK SUITE")
    print("=" * 70)

    part_a = config["part_a"]
    instance_cfg = config["instance"]
    milp_cfg = config["milp"]
    heur_cfg = config["heuristic"]
    part_a_time_limit = float(part_a.get("time_limit", milp_cfg["time_limit"]))

    instances = generate_suite(
        instance_cfg,
        part_a["task_sizes"],
        part_a["qc_counts"],
        int(part_a["num_instances_per_config"]),
        int(part_a.get("seed_base", 42)),
    )
    print(f"Generated {len(instances)} instances")

    rows: List[Dict[str, object]] = []
    eps = 1e-6

    for i, (problem, md) in enumerate(instances, start=1):
        print(f"[{i}/{len(instances)}] Instance {md['instance_id']} ({md['config_id']})")

        milp_result, _ = solve_milp_with_tracking(problem, part_a_time_limit)
        heur_result, _ = solve_heuristic_with_tracking(
            problem,
            time_limit=part_a_time_limit,
            max_iterations=int(heur_cfg["max_iterations"]),
            no_improve_limit=heur_cfg.get("no_improve_limit"),
            seed=int(heur_cfg.get("seed", 42)),
        )

        is_optimal_match = False
        gap_to_opt = None
        if (
            milp_result["status"] == "OPTIMAL"
            and milp_result["objective"] is not None
            and heur_result["objective"] is not None
            and heur_result["is_feasible"]
        ):
            mobj = float(milp_result["objective"])
            hobj = float(heur_result["objective"])
            rel_diff = abs(mobj - hobj) / max(abs(mobj), eps)
            is_optimal_match = rel_diff < eps
            gap_to_opt = abs(hobj - mobj) / max(abs(mobj), eps) * 100.0

        rows.append(
            {
                "instance_id": md["instance_id"],
                "config_id": md["config_id"],
                "num_tasks": md["num_tasks"],
                "num_qcs": md["num_qcs"],
                "num_bays": md["num_bays"],
                "seed": md["seed"],
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
                "gap_to_optimal_percent": format_value(gap_to_opt),
            }
        )

    part_a_path = str(Path(output_dir) / "partA_results.csv")
    part_a_fields = [
        "instance_id",
        "config_id",
        "num_tasks",
        "num_qcs",
        "num_bays",
        "seed",
        "milp_status",
        "milp_objective",
        "milp_best_bound",
        "milp_gap",
        "milp_runtime",
        "milp_node_count",
        "milp_sol_count",
        "heuristic_objective",
        "heuristic_runtime",
        "heuristic_iterations",
        "heuristic_feasible",
        "is_optimal_match",
        "gap_to_optimal_percent",
    ]
    write_csv(part_a_path, rows, part_a_fields)
    write_part_a_summary(rows, str(Path(output_dir) / "partA_summary.csv"))

def write_part_a_summary(rows: List[Dict[str, object]], output_file: str) -> None:
    by_config: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_config[str(r["config_id"])].append(r)

    out_rows: List[Dict[str, object]] = []
    for config_id in sorted(by_config):
        group = by_config[config_id]
        n = len(group)
        optimal_matches = sum(1 for r in group if bool(r["is_optimal_match"]))
        match_rate = optimal_matches / n * 100.0 if n else 0.0

        gaps = []
        for r in group:
            if r["gap_to_optimal_percent"] not in ("N/A", None) and not bool(r["is_optimal_match"]):
                try:
                    gaps.append(float(r["gap_to_optimal_percent"]))
                except Exception:
                    pass

        milp_runtimes = [float(r["milp_runtime"]) for r in group if r["milp_runtime"] != "N/A"]
        heur_runtimes = [float(r["heuristic_runtime"]) for r in group if r["heuristic_runtime"] != "N/A"]

        out_rows.append(
            {
                "config_id": config_id,
                "num_tasks": group[0]["num_tasks"],
                "num_qcs": group[0]["num_qcs"],
                "num_instances": n,
                "optimal_match_count": optimal_matches,
                "optimal_match_rate_percent": format_value(match_rate, 1),
                "mean_gap_to_optimal_percent": format_value(float(np.mean(gaps)) if gaps else None, 2),
                "std_gap_to_optimal_percent": format_value(float(np.std(gaps)) if gaps else None, 2),
                "mean_milp_runtime": format_value(float(np.mean(milp_runtimes)) if milp_runtimes else None, 2),
                "mean_heuristic_runtime": format_value(float(np.mean(heur_runtimes)) if heur_runtimes else None, 2),
            }
        )

    fields = list(out_rows[0].keys()) if out_rows else []
    write_csv(output_file, out_rows, fields)

def run_part_b(config: Dict[str, object], output_dir: str) -> None:
    print("=" * 70)
    print("PART B: PRIMAL BOUND RACE SUITE")
    print("=" * 70)

    part_b = config["part_b"]
    instance_cfg = config["instance"]
    milp_cfg = config["milp"]
    heur_cfg = config["heuristic"]
    part_b_time_limit = float(part_b.get("time_limit", milp_cfg["time_limit"]))
    checkpoints = list(part_b.get("checkpoints", [1, 5, 10, 30, 60, 120, 300, 600]))

    instances = generate_suite(
        instance_cfg,
        part_b["task_sizes"],
        part_b["qc_counts"],
        int(part_b["num_instances_per_config"]),
        int(part_b.get("seed_base", 1000)),
    )
    print(f"Generated {len(instances)} instances")

    run_rows: List[Dict[str, object]] = []
    ts_rows: List[Dict[str, object]] = []

    for i, (problem, md) in enumerate(instances, start=1):
        print(f"[{i}/{len(instances)}] Instance {md['instance_id']} ({md['config_id']})")

        milp_result, milp_trace = solve_milp_with_tracking(problem, part_b_time_limit)
        heur_result, heur_trace = solve_heuristic_with_tracking(
            problem,
            time_limit=part_b_time_limit,
            max_iterations=int(heur_cfg["max_iterations"]),
            no_improve_limit=heur_cfg.get("no_improve_limit"),
            seed=int(heur_cfg.get("seed", 42)),
        )

        milp_cp = extract_checkpoints(milp_trace, checkpoints, "incumbent")
        heur_cp = extract_checkpoints(heur_trace, checkpoints, "best_objective")

        winner = determine_winner(
            float(milp_result["objective"]) if milp_result["objective"] is not None else None,
            float(heur_result["objective"]) if heur_result["objective"] is not None else None,
        )

        gap_milp_heur = None
        if milp_result["objective"] is not None and heur_result["objective"] is not None and heur_result["is_feasible"]:
            hobj = float(heur_result["objective"])
            mobj = float(milp_result["objective"])
            gap_milp_heur = (hobj - mobj) / max(abs(hobj), 1e-6) * 100.0

        row = {
            "instance_id": md["instance_id"],
            "config_id": md["config_id"],
            "num_tasks": md["num_tasks"],
            "num_qcs": md["num_qcs"],
            "num_bays": md["num_bays"],
            "seed": md["seed"],
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
            row[f"milp_obj_at_{cp}s"] = format_value(milp_cp.get(cp))
            row[f"heur_obj_at_{cp}s"] = format_value(heur_cp.get(cp))
        run_rows.append(row)

        for entry in milp_trace:
            ts_rows.append(
                {
                    "instance_id": md["instance_id"],
                    "method": "MILP",
                    "time": format_value(entry.get("time", 0.0)),
                    "objective": format_value(entry.get("incumbent")),
                    "best_bound": format_value(entry.get("best_bound")),
                    "gap": format_value(entry.get("gap")),
                }
            )
        for entry in heur_trace:
            ts_rows.append(
                {
                    "instance_id": md["instance_id"],
                    "method": "HEURISTIC",
                    "time": format_value(entry.get("time", 0.0)),
                    "objective": format_value(entry.get("best_objective")),
                    "best_bound": "N/A",
                    "gap": "N/A",
                }
            )

    run_fields = [
        "instance_id",
        "config_id",
        "num_tasks",
        "num_qcs",
        "num_bays",
        "seed",
        "milp_status",
        "milp_final_obj",
        "milp_best_bound",
        "milp_gap",
        "milp_runtime",
        "milp_node_count",
        "milp_sol_count",
        "heuristic_final_obj",
        "heuristic_runtime",
        "heuristic_iterations",
        "heuristic_feasible",
        "winner",
        "gap_milp_heur_percent",
    ]
    for cp in checkpoints:
        run_fields.extend([f"milp_obj_at_{cp}s", f"heur_obj_at_{cp}s"])

    write_csv(str(Path(output_dir) / "partB_runs.csv"), run_rows, run_fields)
    write_csv(
        str(Path(output_dir) / "partB_timeseries.csv"),
        ts_rows,
        ["instance_id", "method", "time", "objective", "best_bound", "gap"],
    )
    write_part_b_summary(run_rows, str(Path(output_dir) / "partB_summary.csv"))

def write_part_b_summary(rows: List[Dict[str, object]], output_file: str) -> None:
    by_config: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_config[str(r["config_id"])].append(r)

    out_rows: List[Dict[str, object]] = []
    for config_id in sorted(by_config):
        group = by_config[config_id]
        n = len(group)
        winner_counts = {"MILP": 0, "HEURISTIC": 0, "TIE": 0, "NONE": 0}
        for r in group:
            winner_counts[str(r["winner"])] += 1

        milp_runtimes = [float(r["milp_runtime"]) for r in group if r["milp_runtime"] != "N/A"]
        heur_runtimes = [float(r["heuristic_runtime"]) for r in group if r["heuristic_runtime"] != "N/A"]

        gaps = []
        for r in group:
            if r["gap_milp_heur_percent"] not in ("N/A", None):
                try:
                    gaps.append(float(r["gap_milp_heur_percent"]))
                except Exception:
                    pass

        out_rows.append(
            {
                "config_id": config_id,
                "num_tasks": group[0]["num_tasks"],
                "num_qcs": group[0]["num_qcs"],
                "num_instances": n,
                "milp_wins": winner_counts["MILP"],
                "heuristic_wins": winner_counts["HEURISTIC"],
                "ties": winner_counts["TIE"],
                "mean_milp_runtime": format_value(float(np.mean(milp_runtimes)) if milp_runtimes else None, 2),
                "mean_heuristic_runtime": format_value(float(np.mean(heur_runtimes)) if heur_runtimes else None, 2),
                "mean_gap_percent": format_value(float(np.mean(gaps)) if gaps else None, 2),
                "median_gap_percent": format_value(float(np.median(gaps)) if gaps else None, 2),
            }
        )

    fields = list(out_rows[0].keys()) if out_rows else []
    write_csv(output_file, out_rows, fields)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file MILP vs Heuristic benchmark runner")
    parser.add_argument("--suite", choices=["partA", "partB", "both"], default="both")
    parser.add_argument("--output", default="results")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG

    output_dir = args.output
    ensure_dir(output_dir)

    if args.suite in ("partA", "both"):
        run_part_a(config, output_dir)
    if args.suite in ("partB", "both"):
        run_part_b(config, output_dir)

    print("Benchmark complete.")

if __name__ == "__main__":
    main()
