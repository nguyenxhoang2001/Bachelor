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
    """Delay (start, finish) until it doesn't overlap any incompatible task."""
    loops = 0
    while True:
        loops += 1
        if loops > max_loops:
            raise RuntimeError("Non-simultaneity push loop did not converge")

        conflict_finish = None
        for other in incompat_map.get(task, ()):  # only check truly incompatible tasks
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
    """Earliest start/finish on a QC, respecting Φ and Ψ by inserting waiting."""
    travel = _travel_time(problem, qc, prev_task, task)
    pred_ready = 0.0
    for p in predecessors.get(task, ()):  # predecessors must already be scheduled
        pred_ready = max(pred_ready, task_finish[p])

    start = max(qc_ready + travel, pred_ready)
    duration = problem.duration[task]
    finish = start + duration

    # Insert waiting if needed to avoid any incompatible overlap
    start2, finish2 = _push_past_incompatible(task, start, finish, duration, task_intervals, incompat_map)
    return start2, finish2

def solve_greedy(problem):

    start_clock = time.time()

    # -----------------------------
    # Build data structures
    # -----------------------------
    location = problem.location

    # Predecessor map for precedence checking
    predecessors = _build_predecessors(problem.tasks, problem.phi)
    incompat_map = _build_incompatibility_map(problem.psi)

    # -----------------------------
    # Priority ordering
    # -----------------------------
    def priority_key(t: int):
        """Default priority: left→right by bay. If present, prefer discharge+deck."""
        op_rank = 0
        lvl_rank = 0
        if hasattr(problem, "operation"):
            op_rank = 0 if getattr(problem, "operation")[t] == "D" else 1
        if hasattr(problem, "level"):
            lvl_rank = 0 if getattr(problem, "level")[t] == "deck" else 1
        return (location[t], op_rank, lvl_rank)

    unscheduled: List[int] = sorted(problem.tasks, key=priority_key)
    scheduled: Set[int] = set()

    # Track per-task timing as we build (needed for precedence and Ψ-waiting)
    task_finish: Dict[int, float] = {}
    task_intervals: Dict[int, Tuple[float, float]] = {}

    # -----------------------------
    # QC state tracking
    # -----------------------------
    schedule = {k: [] for k in problem.qcs}
    qc_time: Dict[int, float] = {k: float(problem.earliest_time[k]) for k in problem.qcs}
    qc_pos: Dict[int, Optional[int]] = {k: None for k in problem.qcs}

    # Track detailed timing: {qc: [(task, start, finish), ...]}
    qc_task_times: Dict[int, List[Tuple[int, float, float]]] = {k: [] for k in problem.qcs}

    # -----------------------------
    # Main greedy loop
    # -----------------------------
    while unscheduled:
        # Find tasks whose predecessors are already scheduled (so we know their finish times)
        available = [t for t in unscheduled if predecessors.get(t, set()) <= scheduled]

        if not available:
            raise RuntimeError(
                "Precedence deadlock: no available task but unscheduled tasks remain"
            )

        # One-step look-ahead:
        # For each available task, compute its best achievable finish over all QCs,
        # then select the (task, qc) pair that finishes earliest.
        best_task = None
        best_qc = None
        best_start = None
        best_finish = float("inf")

        for task in available:
            for qc in problem.qcs:
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

        # Check if any QC is feasible
        if best_qc is None or best_task is None or best_start is None:
            raise RuntimeError(
                "No feasible assignment found for any available task. "
                f"All assignments violate non-simultaneity constraints."
            )

        # Assign task to best QC
        schedule[best_qc].append(best_task)
        qc_time[best_qc] = best_finish
        qc_pos[best_qc] = best_task
        qc_task_times[best_qc].append((best_task, best_start, best_finish))

        scheduled.add(best_task)
        task_finish[best_task] = best_finish
        task_intervals[best_task] = (best_start, best_finish)

        # Update tracking
        unscheduled.remove(best_task)

    # -----------------------------
    # Calculate objective
    # -----------------------------
    # Re-evaluate the completed schedule using the standard evaluation routine
    # (adds final travel time and uses a consistent timing policy).
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
        "task_times": eval_res["task_times"]  # Detailed timing info
    }

def evaluate_schedule(problem, schedule):
    """Evaluate a schedule (QC -> ordered task list).

    Timing policy (simple + thesis-friendly): Serial Schedule Generation Scheme (SSGS)
    - Each QC executes its list in order (can wait).
    - A task can start only after all predecessors finish (Φ).
    - Incompatible tasks (Ψ) cannot overlap; waiting is inserted as needed.
    - Final travel back to the QC's final position is included in QC completion times.
    """

    tasks = list(problem.tasks)
    qcs = list(problem.qcs)
    predecessors = _build_predecessors(tasks, problem.phi)
    incompat_map = _build_incompatibility_map(problem.psi)

    # Basic validation: every task appears exactly once
    seen: List[int] = []
    for k in qcs:
        seen.extend(schedule.get(k, []))
    if sorted(seen) != sorted(tasks):
        raise ValueError("Schedule must assign each task exactly once")

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

            # Can't schedule until all predecessors have been scheduled (so we know finish time)
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
        

