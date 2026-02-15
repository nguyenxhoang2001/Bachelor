import gurobipy as gp
from gurobipy import GRB


def solve_milp(
    problem,
    time_limit: float = None,
    mip_gap: float = None,
    verbose: bool = False,
    log_file: str = None
):
    from milp_model import build_model, add_constraints, set_objective

    model, X, Y, D, Z, W = build_model(problem)
    add_constraints(model, problem, X, Y, D, Z, W)
    set_objective(model, problem, Y, W)

    if time_limit is not None:
        model.setParam(GRB.Param.TimeLimit, time_limit)

    if mip_gap is not None:
        model.setParam(GRB.Param.MIPGap, mip_gap)

    model.setParam(GRB.Param.LogToConsole, 1 if verbose else 0)
    model.setParam(GRB.Param.Threads, 0)
    if log_file is not None:
        model.setParam(GRB.Param.LogFile, log_file)

    model.optimize()

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD"
    }

    result = {
        "status": status_map.get(model.Status, f"STATUS_{model.Status}"),
        "runtime": model.Runtime,
        "objective": model.ObjVal if model.SolCount > 0 else None,
        "makespan": W.X if model.SolCount > 0 else None,
        "task_completion": {},
        "qc_completion": {},
        "routes": {},
        "best_bound": model.ObjBound,
        "gap": model.MIPGap if model.SolCount > 0 else None,
        "node_count": model.NodeCount,
        "sol_count": model.SolCount
    }

    def _extract_routes(problem, X):
        routes = {}

        for k in problem.qcs:
            route = []
            current = 0
            visited = set()

            while True:
                next_task = None
                for j in problem.tasks + ['T']:
                    if (k, current, j) in X and X[(k, current, j)].X > 0.5:
                        next_task = j
                        break

                if next_task is None or next_task == 'T':
                    break

                if next_task in visited:
                    print(f"WARNING: cycle detected in QC {k}, stopping route extraction")
                    break

                visited.add(next_task)
                route.append(next_task)
                current = next_task

            routes[k] = route

        return routes

    if model.SolCount > 0:

        for i in problem.tasks:
            result["task_completion"][i] = D[i].X

        for k in problem.qcs:
            result["qc_completion"][k] = Y[k].X

        result["routes"] = _extract_routes(problem, X)

    return result
