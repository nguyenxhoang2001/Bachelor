import time
from collections import defaultdict


def solve_greedy(problem):

    start_clock = time.time()

    # -----------------------------
    # Build data structures
    # -----------------------------
    duration = problem.duration
    location = problem.location

    # Predecessor map for precedence checking
    predecessors = {t: set() for t in problem.tasks}
    for i, j in problem.phi:
        predecessors[j].add(i)

    # Successor count: tasks with more successors are scheduled first (critical path)
    successor_count = {t: 0 for t in problem.tasks}
    for i, j in problem.phi:
        successor_count[i] += 1

    # Non-simultaneity pairs
    incompatible = set(problem.psi)

    # -----------------------------
    # Priority ordering
    # -----------------------------
    def priority_key(t):
        """Sort by: more successors first (critical path), then by bay position."""
        return (-successor_count[t], location[t])

    unscheduled = sorted(problem.tasks, key=priority_key)
    completed = set()

    # -----------------------------
    # QC state tracking
    # -----------------------------
    schedule = {k: [] for k in problem.qcs}
    qc_time = {k: problem.earliest_time[k] for k in problem.qcs}
    qc_pos = {k: None for k in problem.qcs}
    
    # Track detailed timing: {qc: [(task, start, finish), ...]}
    qc_task_times = {k: [] for k in problem.qcs}

    # -----------------------------
    # Helper: earliest feasible start avoiding non-simultaneity
    # -----------------------------
    def earliest_feasible_start(task_new, qc_new, earliest):
        """
        Return the earliest start time >= earliest that does not overlap
        with incompatible tasks already scheduled on other QCs.
        Iteratively pushes the start time past any conflicting interval.
        """
        start = earliest
        changed = True
        while changed:
            changed = False
            finish = start + duration[task_new]
            for qc_other in problem.qcs:
                if qc_other == qc_new:
                    continue
                for task_other, start_other, finish_other in qc_task_times[qc_other]:
                    if ((task_new, task_other) in incompatible or
                            (task_other, task_new) in incompatible):
                        if not (finish <= start_other or finish_other <= start):
                            # Delay start to after the conflicting task finishes
                            start = finish_other
                            changed = True
                            break
                if changed:
                    break
        return start

    # -----------------------------
    # Main greedy loop
    # -----------------------------
    while unscheduled:
        # Find tasks with all predecessors completed
        available = [t for t in unscheduled if predecessors[t] <= completed]

        if not available:
            raise RuntimeError(
                "Precedence deadlock: no available task but unscheduled tasks remain"
            )

        # Select the most critical available task (most successors, then leftmost bay)
        task = min(available, key=priority_key)

        # Find best QC for this task
        best_qc = None
        best_finish = float("inf")
        best_start = None

        for qc in problem.qcs:
            # Calculate travel time
            if qc_pos[qc] is None:
                travel = problem.starting_travel_time[(qc, task)]
            else:
                travel = problem.travel_time[(qc, qc_pos[qc], task)]

            # Delay start if needed to satisfy non-simultaneity constraints
            earliest = qc_time[qc] + travel
            start = earliest_feasible_start(task, qc, earliest)
            finish = start + duration[task]

            # Update best if this is earlier
            if finish < best_finish:
                best_finish = finish
                best_qc = qc
                best_start = start

        # Check if any QC is feasible
        if best_qc is None:
            raise RuntimeError(
                f"No feasible QC for task {task} at bay {location[task]}. "
                f"All assignments violate non-simultaneity constraints."
            )

        # Assign task to best QC
        schedule[best_qc].append(task)
        qc_time[best_qc] = best_finish
        qc_pos[best_qc] = task
        qc_task_times[best_qc].append((task, best_start, best_finish))

        # Update tracking
        completed.add(task)
        unscheduled.remove(task)

    # -----------------------------
    # Calculate objective
    # -----------------------------
    makespan = max(qc_time.values())
    total_completion = sum(qc_time.values())
    objective = problem.alpha1 * makespan + problem.alpha2 * total_completion

    runtime = time.time() - start_clock

    return {
        "solver": "Greedy",
        "objective": objective,
        "makespan": makespan,
        "total_completion": total_completion,
        "qc_completion_times": qc_time.copy(),
        "runtime": runtime,
        "status": "FEASIBLE",
        "schedule": schedule,
        "task_times": qc_task_times  # Detailed timing info
    }

def evaluate_schedule(problem, schedule):
    incompatible = set(problem.psi)
    qc_time = {k: problem.earliest_time[k] for k in problem.qcs}
    qc_pos = {k: None for k in problem.qcs}
    qc_task_times = {k: [] for k in problem.qcs}

    def violates_non_simultaneity(task_new, qc_new, start_new, finish_new):
        for qc_other in problem.qcs:
            if qc_other == qc_new:
                continue
            
            for task_other, start_other, finish_other in qc_task_times[qc_other]:
                if ((task_new, task_other) in incompatible or (task_other, task_new) in incompatible):
                
                    if not (finish_new <= start_other or finish_other <= start_new):
                        return True
        return False


    for k in problem.qcs:
        for t in schedule[k]:
            if qc_pos[k] is None:
                travel = problem.starting_travel_time[(k,t)]
            else:
                travel = problem.travel_time[(k,qc_pos[k],t)]

            start = qc_time[k] + travel
            finish = start + problem.duration[t]

            if violates_non_simultaneity(t,k,start,finish):
                raise ValueError
            else:
                qc_time[k] = finish
                qc_pos[k] = t
                qc_task_times[k].append((t,start,finish))


    makespan = max(qc_time.values())
    total_completion = sum(qc_time.values())
    objective = problem.alpha1 * makespan + problem.alpha2 * total_completion

    return {
    "objective": objective,
    "makespan": makespan,
    "total_completion": total_completion,
    "qc_completion_times": qc_time,
    "task_times": qc_task_times,
    "status": "FEASIBLE"}
        

