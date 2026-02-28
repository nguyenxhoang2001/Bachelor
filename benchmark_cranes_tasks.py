from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

from qc_problem import QCProblem
from milp_solver import solve_milp

@dataclass
class ExperimentConfig:
    experiment_type: str
    K: Optional[int] = None
    T: Optional[int] = None
    num_bays: Optional[int] = None
    time_limit: Optional[float] = None
    mip_gap: Optional[float] = None
    seeds: Optional[List[int]] = None
    time_limits: Optional[List[float]] = None

def run_single_instance(
    problem: QCProblem,
    config: ExperimentConfig,
    seed: int,
    time_limit: float,
    mip_gap: Optional[float],
) -> Dict[str, object]:
    result = solve_milp(problem, time_limit=time_limit, mip_gap=mip_gap, verbose=False)
    gap = result.get("gap")
    gap_percent = gap * 100 if gap is not None else None

    return {
        "experiment_type": config.experiment_type,
        "K": config.K,
        "T": config.T,
        "num_bays": config.num_bays,
        "seed": seed,
        "time_limit": time_limit,
        "status": result.get("status"),
        "objective": result.get("objective"),
        "makespan": result.get("makespan"),
        "gap_percent": gap_percent,
        "runtime": result.get("runtime"),
        "best_bound": result.get("best_bound"),
        "node_count": result.get("node_count"),
        "sol_count": result.get("sol_count"),
    }

def write_results_csv(
    file_path: str,
    rows: Iterable[Dict[str, object]],
    fieldnames: List[str],
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    with open(file_path, "w", newline="") as f:
        if metadata:
            for k, v in metadata.items():
                f.write(f"# {k}: {v}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def run_task_scaling(
    seeds: List[int],
    time_limit: float,
    output_csv: str,
) -> None:
    rows: List[Dict[str, object]] = []

    Ks = [2, 3]
    Ts = [5, 10, 15, 20]

    for K in Ks:
        for T in Ts:
            num_bays = T
            config = ExperimentConfig(
                experiment_type="task_scaling",
                K=K,
                T=T,
                num_bays=num_bays,
                time_limit=time_limit,
                mip_gap=None,
                seeds=seeds,
            )
            for seed in seeds:
                problem = QCProblem.generate(
                    num_bays=num_bays,
                    num_qcs=K,
                    total_tasks=T,
                    seed=seed,
                )
                row = run_single_instance(problem, config, seed, time_limit, None)
                rows.append(row)

    metadata = {
        "experiment_type": "task_scaling",
        "Ks": Ks,
        "Ts": Ts,
        "num_bays_rule": "equal_to_T",
        "time_limit": time_limit,
        "mip_gap": None,
        "seeds": seeds,
    }
    fieldnames = [
        "experiment_type",
        "K",
        "T",
        "num_bays",
        "seed",
        "time_limit",
        "status",
        "objective",
        "makespan",
        "gap_percent",
        "runtime",
        "best_bound",
        "node_count",
        "sol_count",
    ]
    write_results_csv(output_csv, rows, fieldnames, metadata)

def run_capacity_test(
    Ks: List[int],
    Ts: List[int],
    seeds: List[int],
    time_limit: float,
    output_csv: str,
) -> None:
    rows: List[Dict[str, object]] = []
    num_bays = 30

    for K in Ks:
        for T in Ts:
            config = ExperimentConfig(
                experiment_type="capacity_test",
                K=K,
                T=T,
                num_bays=num_bays,
                time_limit=time_limit,
                mip_gap=None,
                seeds=seeds,
            )
            for seed in seeds:
                problem = QCProblem.generate(
                    num_bays=num_bays,
                    num_qcs=K,
                    total_tasks=T,
                    seed=seed,
                )
                row = run_single_instance(problem, config, seed, time_limit, None)
                rows.append(row)

    metadata = {
        "experiment_type": "capacity_test",
        "Ks": Ks,
        "Ts": Ts,
        "num_bays": num_bays,
        "time_limit": time_limit,
        "mip_gap": None,
        "seeds": seeds,
    }
    fieldnames = [
        "experiment_type",
        "K",
        "T",
        "num_bays",
        "seed",
        "time_limit",
        "status",
        "objective",
        "makespan",
        "gap_percent",
        "runtime",
        "best_bound",
        "node_count",
        "sol_count",
    ]
    write_results_csv(output_csv, rows, fieldnames, metadata)

def run_time_limit(
    K: int,
    T: int,
    num_bays: int,
    seeds: List[int],
    time_limits: List[float],
    output_csv: str,
) -> None:
    rows: List[Dict[str, object]] = []

    problems: Dict[int, QCProblem] = {
        seed: QCProblem.generate(
            num_bays=num_bays,
            num_qcs=K,
            total_tasks=T,
            seed=seed,
        )
        for seed in seeds
    }

    for seed, problem in problems.items():
        config = ExperimentConfig(
            experiment_type="time_limit",
            K=K,
            T=T,
            num_bays=num_bays,
            time_limit=None,
            mip_gap=None,
            seeds=seeds,
            time_limits=time_limits,
        )
        for tl in time_limits:
            row = run_single_instance(problem, config, seed, tl, None)
            rows.append(row)

    metadata = {
        "experiment_type": "time_limit",
        "K": K,
        "T": T,
        "num_bays": num_bays,
        "time_limits": time_limits,
        "mip_gap": None,
        "seeds": seeds,
    }
    fieldnames = [
        "experiment_type",
        "K",
        "T",
        "num_bays",
        "seed",
        "time_limit",
        "status",
        "objective",
        "makespan",
        "gap_percent",
        "runtime",
        "best_bound",
        "node_count",
        "sol_count",
    ]
    write_results_csv(output_csv, rows, fieldnames, metadata)

def main() -> None:
    fixed_seeds = list(range(1, 6))

    '''
    #1. task scaling
    run_task_scaling(
        seeds=fixed_seeds,
        time_limit=1000.0,
        output_csv="task_scaling.csv",
    )
    '''
    #2. capacity test
    '''
    
    run_capacity_test(
        Ks=[2, 3],
        Ts=[5, 6, 7, 8, 9, 10, 11, 12],
        seeds=fixed_seeds,
        time_limit=600.0,
        output_csv="capacity_test.csv",
    )
'''
    # 3. time-limit/anytime curve
    
    run_time_limit(
        K=3,
        T=10,
        num_bays=11,
        seeds=fixed_seeds,
        time_limits=[1000, 1500, 2000, 2500],
        output_csv="time_limit.csv",
    )

if __name__ == "__main__":
    main()
