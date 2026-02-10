from qc_problem import QCProblem
from milp_solver import solve_milp
from greedysolver import solve_greedy


def run_small_test():
    num_bays = 7
    num_qcs = 3
    total_tasks = 6
    seed = 67

    print("\n" + "=" * 70)
    print("QC SCHEDULING – SMALL TEST INSTANCE")
    print("=" * 70)

    problem = QCProblem.generate(
        num_bays=num_bays,
        num_qcs=num_qcs,
        total_tasks=total_tasks,
        seed=seed
    )

    print("\nINSTANCE OVERVIEW")
    print("-" * 70)
    print(f"Bays:        {num_bays}")
    print(f"QCs:         {num_qcs}")
    print(f"Tasks:       {total_tasks}")
    print(f"Random seed: {seed}")

    print("\nTask locations (task → bay):")
    for t in problem.tasks:
        print(f"  Task {t:2d} → Bay {problem.location[t]}")

    print("\nACTIVE CONSTRAINTS")
    print("-" * 70)
    print(f"Precedence constraints Φ:       {len(problem.phi)}")
    print(f"Non-simultaneity constraints Ψ: {len(problem.psi)}")

   
    milp_result = solve_milp(
        problem,
        time_limit=600,
        mip_gap=0.05,
        verbose=False
    )

    greedy_result = solve_greedy(problem)

    print("\nRESULTS SUMMARY")
    print("-" * 70)
    print(f"{'Metric':<18} {'MILP':<15} {'Greedy':<15}")
    print("-" * 70)

    print(f"{'Status':<18} {milp_result['status']:<15} {greedy_result['status']:<15}")
    print(f"{'Objective':<18} {milp_result['objective']:<15.2f} {greedy_result['objective']:<15.2f}")
    print(f"{'Makespan':<18} {milp_result['makespan']:<15.2f} {greedy_result['makespan']:<15.2f}")
    print(f"{'Runtime (s)':<18} {milp_result['runtime']:<15.4f} {greedy_result['runtime']:<15.4f}")

    print("\nQC SCHEDULES")
    print("-" * 70)

    print("\nMILP:")
    if milp_result.get("routes"):
        for k in problem.qcs:
            print(f"  QC {k}: {milp_result['routes'][k]}")
    else:
        print("  No MILP schedule available")

    print("\nGreedy:")
    for k in problem.qcs:
        print(f"  QC {k}: {greedy_result['schedule'][k]}")

    print("\nCOMPARISON")
    print("-" * 70)

    # Objective difference
    obj_diff = greedy_result["objective"] - milp_result["objective"]
    obj_pct = (obj_diff / milp_result["objective"]) * 100 if milp_result["objective"] > 0 else 0
    print(f"Objective difference (Greedy − MILP): {obj_diff:+.2f} ({obj_pct:+.2f}%)")

    # Runtime difference
    milp_time = milp_result["runtime"]
    greedy_time = greedy_result["runtime"]
    time_diff = milp_time - greedy_time

    if greedy_time > 1e-6:
        speedup = milp_time / greedy_time
        speedup_str = f"{speedup:.1f}×"
    else:
        speedup_str = "∞"

    print(f"Runtime difference (MILP − Greedy):   {time_diff:.4f} s")
    print(f"Runtime speedup (MILP / Greedy):      {speedup_str}")

    # MILP gap
    milp_gap = milp_result.get("gap", None)
    if milp_gap is not None:
        print(f"MILP optimality gap:                  {milp_gap * 100:.2f} %")
    else:
        print(f"MILP optimality gap:                  N/A")

    print("\n" + "=" * 70)
    print("END OF TEST")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_small_test()
