from __future__ import annotations

import time
import csv
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

from qc_problem import QCProblem
from milp_solver import solve_milp

@dataclass
class ResultSummary:
    K: int
    T: int
    n_instances: int
    avg_obj: float
    avg_gap: float
    avg_time: float
    opt_count: int
    tl_count: int

def benchmark_configuration(
    K: int,
    T: int,
    seeds: List[int],
    time_limit: float,
    max_bay: int | None = None,
    mip_gap: float | None = None,
) -> ResultSummary:
    objs: List[float] = []
    gaps: List[float] = []
    times: List[float] = []
    opt_count = 0
    tl_count = 0
    n_instances = len(seeds)

    for seed in seeds:
        num_bays = max_bay if max_bay is not None else T
        problem = QCProblem.generate(
            num_bays=num_bays,
            num_qcs=K,
            total_tasks=T,
            seed=seed,
        )
        result = solve_milp(
            problem,
            time_limit=time_limit,
            mip_gap=mip_gap,
            verbose=False
        )
        runtime = result.get("runtime")
        times.append(runtime if runtime is not None else float('nan'))
        makespan = result.get("makespan")
        status_str = result.get("status")
        gap = result.get("gap")

        if makespan is not None:
            objs.append(makespan)
        else:
            objs.append(float('nan'))

        if status_str == "OPTIMAL":
            gap = 0.0
        gaps.append(gap)

        if status_str == "OPTIMAL":
            opt_count += 1
        elif status_str == "TIME_LIMIT":
            tl_count += 1

    def safe_mean(lst: List[float]) -> float:
        vals = [x for x in lst if x == x]  # filter NaNs
        return sum(vals) / len(vals) if vals else float('nan')

    return ResultSummary(
        K=K,
        T=T,
        n_instances=n_instances,
        avg_obj=safe_mean(objs),
        avg_gap=safe_mean(gaps),
        avg_time=safe_mean(times),
        opt_count=opt_count,
        tl_count=tl_count,
    )


def _write_rows_csv(file_path: str, fieldnames: List[str], rows: Iterable[Dict[str, object]]) -> None:
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def time_limit_curve(
    K: int,
    T: int,
    num_bays: int,
    seeds: List[int],
    time_limits: List[float],
    mip_gap: float | None = None,
    output_csv: str = "milp_time_limit_curve.csv",
) -> None:
    # Use identical instances across all time limits for anytime curves.
    problems = {
        seed: QCProblem.generate(
            num_bays=num_bays,
            num_qcs=K,
            total_tasks=T,
            seed=seed,
        )
        for seed in seeds
    }

    rows: List[Dict[str, object]] = []
    for time_limit in time_limits:
        for seed, problem in problems.items():
            result = solve_milp(
                problem,
                time_limit=time_limit,
                mip_gap=mip_gap,
                verbose=False,
            )
            rows.append({
                "K": K,
                "T": T,
                "num_bays": num_bays,
                "seed": seed,
                "time_limit": time_limit,
                "status": result.get("status"),
                "objective": result.get("objective"),
                "makespan": result.get("makespan"),
                "gap": result.get("gap"),
                "runtime": result.get("runtime"),
                "best_bound": result.get("best_bound"),
                "node_count": result.get("node_count"),
                "sol_count": result.get("sol_count"),
            })

    _write_rows_csv(
        output_csv,
        [
            "K",
            "T",
            "num_bays",
            "seed",
            "time_limit",
            "status",
            "objective",
            "makespan",
            "gap",
            "runtime",
            "best_bound",
            "node_count",
            "sol_count",
        ],
        rows,
    )


def bay_density_benchmark(
    K: int,
    T: int,
    bay_values: List[int],
    seeds: List[int],
    time_limit: float,
    mip_gap: float | None = None,
    output_csv: str = "milp_bay_density.csv",
) -> None:
    rows: List[Dict[str, object]] = []

    for num_bays in bay_values:
        for seed in seeds:
            problem = QCProblem.generate(
                num_bays=num_bays,
                num_qcs=K,
                total_tasks=T,
                seed=seed,
            )
            result = solve_milp(
                problem,
                time_limit=time_limit,
                mip_gap=mip_gap,
                verbose=False,
            )
            rows.append({
                "K": K,
                "T": T,
                "num_bays": num_bays,
                "seed": seed,
                "time_limit": time_limit,
                "status": result.get("status"),
                "objective": result.get("objective"),
                "makespan": result.get("makespan"),
                "gap": result.get("gap"),
                "runtime": result.get("runtime"),
                "best_bound": result.get("best_bound"),
                "node_count": result.get("node_count"),
                "sol_count": result.get("sol_count"),
            })

    _write_rows_csv(
        output_csv,
        [
            "K",
            "T",
            "num_bays",
            "seed",
            "time_limit",
            "status",
            "objective",
            "makespan",
            "gap",
            "runtime",
            "best_bound",
            "node_count",
            "sol_count",
        ],
        rows,
    )

def main():
    task_values: List[int] = [5, 10, 15, 20]
    crane_values: List[int] = [2, 3]
    n_instances = 5
    seeds = list(range(1, n_instances + 1))

    time_limit_map = {
        5: 2000.0,
        10: 1000.0,
        15: 1500.0,
        20: 2000.0,
    }
    mip_gap_map = {
        5: None,
        10: None,
        15: None,
        20: None
    }
    bay_count = 30
    results: List[ResultSummary] = []

    for K in crane_values:
        for T in task_values:
            summary = benchmark_configuration(
                K=K,
                T=T,
                seeds=seeds,
                time_limit=time_limit_map[T],
                max_bay=bay_count,
                mip_gap=mip_gap_map[T],
            )
            results.append(summary)

    header = (
        f"{'K':<3} | {'T':<3} | {'Inst':>5} | {'Avg Obj':>12} | "
        f"{'Avg Gap (%)':>11} | {'Avg Time (s)':>12} | {'Opt':>5} | {'TL':>5}"
    )
    print(header)
    print('-' * len(header))
    for res in results:
        avg_obj_str = f"{res.avg_obj:.2f}" if res.avg_obj == res.avg_obj else "nan"
        avg_gap_str = f"{res.avg_gap:.2f}" if res.avg_gap == res.avg_gap else "nan"
        avg_time_str = f"{res.avg_time:.2f}" if res.avg_time == res.avg_time else "nan"
        print(
            f"{res.K:<3} | {res.T:<3} | {res.n_instances:>5} | {avg_obj_str:>12} | "
            f"{avg_gap_str:>11} | {avg_time_str:>12} | {res.opt_count:>5} | {res.tl_count:>5}"
        )

    with open("milp_benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "K", "T", "n_instances", "avg_obj", "avg_gap", "avg_time", "opt_count", "tl_count"
        ])
        writer.writeheader()
        for res in results:
            writer.writerow(asdict(res))

    run_time_limit_curve = False
    run_bay_density = False

    if run_time_limit_curve:
        time_limit_curve(
            K=3,
            T=15,
            num_bays=30,
            seeds=seeds,
            time_limits=[50, 100, 200, 500, 1000],
            mip_gap=None,
            output_csv="milp_time_limit_curve.csv",
        )

    if run_bay_density:
        bay_density_benchmark(
            K=3,
            T=15,
            bay_values=[10, 20, 30, 40, 60],
            seeds=seeds,
            time_limit=1000.0,
            mip_gap=None,
            output_csv="milp_bay_density.csv",
        )

if __name__ == '__main__':
    main()
