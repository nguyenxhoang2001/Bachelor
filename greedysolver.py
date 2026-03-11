import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _build_predecessors(tasks: Iterable[int], phi: Iterable[Tuple[int, int]]) -> Dict[int, Set[int]]:
    predecessors: Dict[int, Set[int]] = {t: set() for t in tasks}
    for i, j in phi:
        predecessors.setdefault(j, set()).add(i)
        predecessors.setdefault(i, set())
    return predecessors

def _build_incompatibility_map(psi: Iterable[Tuple[int, int]]) -> Dict[int, Set[int]]:
    inc: Dict[int, Set[int]] = {}
    for i, j in psi:
        inc.setdefault(i, set()).add(j)
        inc.setdefault(j, set()).add(i)
    return inc

def check_interference(problem, schedule) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
    qc_of_task: Dict[int, int] = {}
    for qc, tasks in schedule.items():
        for t in tasks:
            qc_of_task[t] = qc

    tasks = list(problem.tasks)
    for i in tasks:
        for j in tasks:
            if problem.location[i] < problem.location[j]:
                qc_i = qc_of_task.get(i)
                qc_j = qc_of_task.get(j)
                if qc_i is None or qc_j is None:
                    return False, (i, j, qc_i or -1, qc_j or -1)
                if qc_i > qc_j:
                    return False, (i, j, qc_i, qc_j)

    return True, None

def _travel_time(problem, qc: int, prev_task: Optional[int], task: int) -> float:
    if prev_task is None:
        return problem.starting_travel_time[(qc, task)]
    return problem.travel_time[(qc, prev_task, task)]

def _push_past_incompatible(
    task: int,
    start: float,
    finish: float,
    duration: float,
    task_intervals: Dict[int, Tuple[float, float]],
    incompat_map: Dict[int, Set[int]],
    max_loops: int = 10_000,
) -> Tuple[float, float]:
    loops = 0
    while True:
        loops += 1
        if loops > max_loops:
            raise RuntimeError("Non-simultaneity push loop did not converge")

        conflict_finish = None
        for other in incompat_map.get(task, ()):
            if other not in task_intervals:
                continue
            other_start, other_finish = task_intervals[other]
            if not (finish <= other_start or other_finish <= start):
                conflict_finish = other_finish if conflict_finish is None else max(conflict_finish, other_finish)

        if conflict_finish is None:
            return start, finish

        start = conflict_finish
        finish = start + duration

def _earliest_feasible_start(
    problem,
    qc: int,
    task: int,
    qc_ready: float,
    prev_task: Optional[int],
    predecessors: Dict[int, Set[int]],
    task_finish: Dict[int, float],
    task_intervals: Dict[int, Tuple[float, float]],
    incompat_map: Dict[int, Set[int]],
) -> Tuple[float, float]:
    travel = _travel_time(problem, qc, prev_task, task)
    pred_ready = 0.0
    for p in predecessors.get(task, ()):
        pred_ready = max(pred_ready, task_finish[p])

    start = max(qc_ready + travel, pred_ready)
    duration = problem.duration[task]
    finish = start + duration

    start2, finish2 = _push_past_incompatible(task, start, finish, duration, task_intervals, incompat_map)
    return start2, finish2

def solve_greedy(problem):

    start_clock = time.time()
    location = problem.location
    predecessors = _build_predecessors(problem.tasks, problem.phi)
    incompat_map = _build_incompatibility_map(problem.psi)
    assigned_qc: Dict[int, int] = {}

    unscheduled: List[int] = sorted(problem.tasks, key=lambda t: location[t])
    scheduled: Set[int] = set()

    task_finish: Dict[int, float] = {}
    task_intervals: Dict[int, Tuple[float, float]] = {}

    schedule = {k: [] for k in problem.qcs}
    qc_time: Dict[int, float] = {k: float(problem.earliest_time[k]) for k in problem.qcs}
    qc_pos: Dict[int, Optional[int]] = {k: None for k in problem.qcs}

    qc_task_times: Dict[int, List[Tuple[int, float, float]]] = {k: [] for k in problem.qcs}

    while unscheduled:
        available = [t for t in unscheduled if predecessors.get(t, set()) <= scheduled]

        if not available:
            raise RuntimeError(
                "Precedence deadlock: no available task but unscheduled tasks remain"
            )

        best_task = None
        best_qc = None
        best_start = None
        best_finish = float("inf")

        for task in available:
            lower_qc = min(problem.qcs)
            upper_qc = max(problem.qcs)
            for t, qc in assigned_qc.items():
                if location[t] < location[task]:
                    lower_qc = max(lower_qc, qc)
                elif location[t] > location[task]:
                    upper_qc = min(upper_qc, qc)

            if lower_qc > upper_qc:
                continue

            for qc in problem.qcs:
                if qc < lower_qc or qc > upper_qc:
                    continue
                start, finish = _earliest_feasible_start(
                    problem,
                    qc,
                    task,
                    qc_time[qc],
                    qc_pos[qc],
                    predecessors,
                    task_finish,
                    task_intervals,
                    incompat_map,
                )
                if finish < best_finish:
                    best_finish = finish
                    best_qc = qc
                    best_task = task
                    best_start = start

        if best_qc is None or best_task is None or best_start is None:
            raise RuntimeError(
                "No feasible assignment found for any available task. "
                "Assignments may violate non-simultaneity or interference constraints."
            )

        schedule[best_qc].append(best_task)
        assigned_qc[best_task] = best_qc
        qc_time[best_qc] = best_finish
        qc_pos[best_qc] = best_task
        qc_task_times[best_qc].append((best_task, best_start, best_finish))

        scheduled.add(best_task)
        task_finish[best_task] = best_finish
        task_intervals[best_task] = (best_start, best_finish)

        unscheduled.remove(best_task)

    eval_res = evaluate_schedule(problem, schedule)

    runtime = time.time() - start_clock

    return {
        "solver": "Greedy",
        "objective": eval_res["objective"],
        "makespan": eval_res["makespan"],
        "total_completion": eval_res["total_completion"],
        "qc_completion_times": eval_res["qc_completion_times"].copy(),
        "runtime": runtime,
        "status": "FEASIBLE",
        "schedule": schedule,
        "task_times": eval_res["task_times"]
    }

def evaluate_schedule(problem, schedule):
    tasks = list(problem.tasks)
    qcs = list(problem.qcs)
    predecessors = _build_predecessors(tasks, problem.phi)
    incompat_map = _build_incompatibility_map(problem.psi)

    seen: List[int] = []
    for k in qcs:
        seen.extend(schedule.get(k, []))
    if sorted(seen) != sorted(tasks):
        raise ValueError("Schedule must assign each task exactly once")

    ok, violation = check_interference(problem, schedule)
    if not ok:
        raise ValueError(f"Interference constraint violated: {violation}")

    qc_idx: Dict[int, int] = {k: 0 for k in qcs}
    qc_ready: Dict[int, float] = {k: float(problem.earliest_time[k]) for k in qcs}
    qc_prev: Dict[int, Optional[int]] = {k: None for k in qcs}
    qc_task_times: Dict[int, List[Tuple[int, float, float]]] = {k: [] for k in qcs}

    task_finish: Dict[int, float] = {}
    task_intervals: Dict[int, Tuple[float, float]] = {}

    remaining = len(tasks)
    while remaining > 0:
        best_k = None
        best_t = None
        best_start = None
        best_finish = None

        for k in qcs:
            idx = qc_idx[k]
            if idx >= len(schedule[k]):
                continue
            t = schedule[k][idx]

            if any(p not in task_finish for p in predecessors.get(t, ())):
                continue

            start, finish = _earliest_feasible_start(
                problem,
                k,
                t,
                qc_ready[k],
                qc_prev[k],
                predecessors,
                task_finish,
                task_intervals,
                incompat_map,
            )
            if best_start is None or start < best_start or (start == best_start and finish < (best_finish or float("inf"))):
                best_k = k
                best_t = t
                best_start = start
                best_finish = finish

        if best_k is None or best_t is None or best_start is None or best_finish is None:
            raise ValueError("Precedence deadlock: schedule order is infeasible w.r.t. Φ")

        qc_task_times[best_k].append((best_t, best_start, best_finish))
        qc_ready[best_k] = best_finish
        qc_prev[best_k] = best_t
        qc_idx[best_k] += 1

        task_finish[best_t] = best_finish
        task_intervals[best_t] = (best_start, best_finish)
        remaining -= 1

    qc_completion: Dict[int, float] = {}
    for k in qcs:
        last = qc_prev[k]
        if last is None:
            qc_completion[k] = float(problem.earliest_time[k])
        else:
            qc_completion[k] = qc_ready[k] + float(problem.final_travel_time.get((k, last), 0.0))

    makespan = max(qc_completion.values()) if qc_completion else 0.0
    total_completion = sum(qc_completion.values())
    objective = problem.alpha1 * makespan + problem.alpha2 * total_completion

    return {
        "objective": objective,
        "makespan": makespan,
        "total_completion": total_completion,
        "qc_completion_times": qc_completion,
        "task_times": qc_task_times,
        "status": "FEASIBLE",
    }
        

