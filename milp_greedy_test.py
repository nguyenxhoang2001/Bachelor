import csv
import time
from qc_problem import QCProblem
from milp_solver import solve_milp
from greedysolver import solve_greedy


def run_greedy_vs_milp_benchmark(
    task_sizes,
    num_bays=7,
    num_qcs=3,
    seed=42,
    time_limit=60,
    mip_gap=0.05,
    output_csv="greedy_vs_milp_results.csv"
):
    print("\n" + "=" * 90)
    print("GREEDY vs MILP BENCHMARK (TIME-LIMITED MILP)")
    print("=" * 90)

    print("\nFixed parameters:")
    print(f"  Bays:        {num_bays}")
    print(f"  QCs:         {num_qcs}")
    print(f"  Seed:        {seed}")
    print(f"  MILP limit:  {time_limit} s")
    print(f"  MILP gap:    {mip_gap}")

    print("\n" + "-" * 90)
    print(
        f"{'Tasks':<6} "
        f"{'MILP Obj':<12} {'Greedy Obj':<12} "
        f"{'Δ Obj (%)':<10} "
        f"{'MILP Gap (%)':<12} "
        f"{'MILP Time (s)':<14} {'Greedy Time (s)':<16}"
    )
    print("-" * 90)

    rows = []

    for n_tasks in task_sizes:
        # -------------------------------
        # Generate identical instance
        # -------------------------------
        problem = QCProblem.generate(
            num_bays=num_bays,
            num_qcs=num_qcs,
            total_tasks=n_tasks,
            seed=seed
        )

        # -------------------------------
        # MILP
        # -------------------------------
        milp_start = time.time()
        milp_result = solve_milp(
            problem,
            time_limit=time_limit,
            mip_gap=mip_gap,
            verbose=False
        )
        milp_time = time.time() - milp_start

        milp_obj = milp_result.get("objective", None)
        milp_makespan = milp_result.get("makespan", None)
        milp_gap = milp_result.get("gap", None)
        milp_gap_pct = milp_gap * 100 if milp_gap is not None else None

        # -------------------------------
        # Greedy
        # -------------------------------
        greedy_start = time.time()
        greedy_result = solve_greedy(problem)
        greedy_time = time.time() - greedy_start

        greedy_obj = greedy_result.get("objective", None)
        greedy_makespan = greedy_result.get("makespan", None)

        # -------------------------------
        # Comparison
        # -------------------------------
        if milp_obj is not None and greedy_obj is not None:
            obj_diff_pct = ((greedy_obj - milp_obj) / milp_obj) * 100
        else:
            obj_diff_pct = None

        print(
            f"{n_tasks:<6} "
            f"{milp_obj if milp_obj is not None else 'N/A':<12.2f} "
            f"{greedy_obj if greedy_obj is not None else 'N/A':<12.2f} "
            f"{obj_diff_pct if obj_diff_pct is not None else 'N/A':<10.2f} "
            f"{milp_gap_pct if milp_gap_pct is not None else 'N/A':<12.2f} "
            f"{milp_time:<14.2f} "
            f"{greedy_time:<16.6f}"
        )

        rows.append({
            "tasks": n_tasks,
            "bays": num_bays,
            "qcs": num_qcs,
            "seed": seed,
            "milp_time_limit": time_limit,
            "milp_objective": milp_obj,
            "milp_makespan": milp_makespan,
            "milp_runtime_sec": milp_time,
            "milp_gap": milp_gap,
            "milp_gap_percent": milp_gap_pct,
            "greedy_objective": greedy_obj,
            "greedy_makespan": greedy_makespan,
            "greedy_runtime_sec": greedy_time,
            "objective_diff_percent": obj_diff_pct
        })

    # -------------------------------
    # Write CSV
    # -------------------------------
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "-" * 90)
    print(f"Results written to: {output_csv}")
    print("This file can be used directly for tables and plots.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    TASK_SIZES = [6, 8, 10, 12, 15, 20]

    run_greedy_vs_milp_benchmark(
        task_sizes=TASK_SIZES,
        num_bays=7,
        num_qcs=3,
        seed=42,
        time_limit=60,      # or 300 for extended comparison
        mip_gap=0.05
    )
