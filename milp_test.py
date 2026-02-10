import os
import csv
import gurobipy as gp
from gurobipy import GRB

from qc_problem import QCProblem
from milp_model import build_model, add_constraints, set_objective


# ------------------------------------------------------------
# 1) Solve and log primal objective improvements over time
#    + Save the full Gurobi text log to a .log file
# ------------------------------------------------------------
def solve_milp_primal_over_time(problem, time_limit_sec, out_csv, log_file=None, verbose=False):
    model, X, Y, D, Z, W = build_model(problem)
    add_constraints(model, problem, X, Y, D, Z, W)
    set_objective(model, problem, Y, W)

    model.setParam(GRB.Param.TimeLimit, time_limit_sec)
    model.setParam(GRB.Param.Threads, 1)

    # --- NEW: log routing ---
    # If you're saving to file, usually keep console quiet unless verbose=True
    model.setParam(GRB.Param.LogToConsole, 1 if verbose else 0)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        model.setParam(GRB.Param.LogFile, log_file)
        # Overwrite each time (avoid accidental appends across runs)
        model.setParam(GRB.Param.AppendToLogFile, 0)

    # Store (time_sec, incumbent_obj) whenever a new incumbent is found
    trace = []

    def cb(m, where):
        if where == GRB.Callback.MIPSOL:
            t = m.cbGet(GRB.Callback.RUNTIME)
            obj = m.cbGet(GRB.Callback.MIPSOL_OBJ)
            trace.append((t, obj))

    model.optimize(cb)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "primal_objective"])
        for t, obj in trace:
            w.writerow([t, obj])

    return {
        "status_code": model.Status,
        "status": _status_to_str(model.Status),
        "runtime": model.Runtime,
        "sol_count": model.SolCount,
        "final_objective": model.ObjVal if model.SolCount > 0 else None,
        "out_csv": out_csv,
        "log_file": log_file
    }


def _status_to_str(code):
    mapping = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
    }
    return mapping.get(code, f"STATUS_{code}")


# ------------------------------------------------------------
# 2) Build a step-function f(t) from a primal log CSV
# ------------------------------------------------------------
def build_primal_step_function_from_csv(csv_path):
    """
    Returns a function f(t) that outputs the best (lowest) primal objective
    found up to time t (seconds). If no incumbent exists yet, returns None.
    """
    points = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            t = float(row["time_sec"])
            obj = float(row["primal_objective"])
            points.append((t, obj))

    points.sort(key=lambda x: x[0])

    best = None
    best_points = []
    for t, obj in points:
        if best is None or obj < best:
            best = obj
            best_points.append((t, best))

    def f(t_query):
        if not best_points:
            return None
        last = None
        for t, val in best_points:
            if t <= t_query:
                last = val
            else:
                break
        return last

    return f, best_points


def load_all_primal_functions(log_dir, task_sizes):
    out = {}
    for n in task_sizes:
        csv_path = os.path.join(log_dir, f"primal_over_time_t{n}.csv")
        f, pts = build_primal_step_function_from_csv(csv_path)
        out[n] = {"csv": csv_path, "f": f, "points": pts}
    return out


# ------------------------------------------------------------
# 3) Main test runner: tasks 10..60 by 10, time limit 2000s
# ------------------------------------------------------------
def run_primal_benchmark(
    task_sizes=range(10, 61, 10),
    num_bays=7,
    num_qcs=3,
    seed=42,
    time_limit_sec=1800,
    log_dir="primal_logs",
    verbose=False
):
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print("MILP PRIMAL OVER TIME BENCHMARK (INCUMBENT TRACE)")
    print("=" * 90)
    print(f"Task sizes: {list(task_sizes)}")
    print(f"Bays: {num_bays}, QCs: {num_qcs}, Seed: {seed}, TimeLimit: {time_limit_sec}s")
    print("-" * 90)

    summary_rows = []

    for n_tasks in task_sizes:
        print(f"\n--- Solving instance: tasks={n_tasks} ---")

        problem = QCProblem.generate(
            num_bays=num_bays,
            num_qcs=num_qcs,
            total_tasks=n_tasks,
            seed=seed
        )

        out_csv = os.path.join(log_dir, f"primal_over_time_t{n_tasks}.csv")

        # --- NEW: per-instance gurobi log file ---
        out_log = os.path.join(log_dir, f"gurobi_t{n_tasks}.log")

        res = solve_milp_primal_over_time(
            problem=problem,
            time_limit_sec=time_limit_sec,
            out_csv=out_csv,
            log_file=out_log,
            verbose=verbose
        )

        print(f"Status: {res['status']}, Runtime: {res['runtime']:.2f}s, "
              f"Solutions: {res['sol_count']}, FinalObj: {res['final_objective']}")
        print(f"Saved: {res['out_csv']} and {res['log_file']}")

        summary_rows.append({
            "tasks": n_tasks,
            "bays": num_bays,
            "qcs": num_qcs,
            "seed": seed,
            "time_limit_sec": time_limit_sec,
            "status": res["status"],
            "runtime_sec": res["runtime"],
            "sol_count": res["sol_count"],
            "final_objective": res["final_objective"],
            "log_csv": out_csv,
            "gurobi_log": out_log
        })

    summary_csv = os.path.join(log_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)

    print("\n" + "-" * 90)
    print("Summary written to:", summary_csv)
    print("Per-instance primal traces and Gurobi logs are in:", log_dir)
    print("=" * 90)

    primal_funcs = load_all_primal_functions(log_dir, list(task_sizes))
    return primal_funcs


if __name__ == "__main__":
    run_primal_benchmark(
        task_sizes=range(10, 61, 10),
        num_bays=7,
        num_qcs=3,
        seed=42,
        time_limit_sec=1800,
        log_dir="primal_logs",
        verbose=False
    )
