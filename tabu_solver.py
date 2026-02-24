from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from greedysolver import evaluate_schedule, solve_greedy


Schedule = Dict[int, List[int]]


def _schedule_key(problem, schedule: Schedule) -> Tuple[Tuple[int, ...], ...]:
    qcs = sorted(problem.qcs)
    return tuple(tuple(schedule[k]) for k in qcs)

def solve_tabu(
    problem,
    max_iteration: int = 50,
    tabu_tenure: Optional[int] = None,
    tenure_jitter: int = 3,
    move_types: Sequence[str] = ("relocate", "swap"),
    max_neighbors: Optional[int] = None,
    seed: Optional[int] = None,
):
    """Simple tabu search that improves a greedy initial schedule.

    - Neighborhoods: relocation with insertion + swap between QCs.
    - Tabu: store moves with an expiration iteration; forbid immediate reversal.
    - Aspiration: allow tabu moves if they improve the global best objective.
    - Speed: memoize schedule evaluations.
    """

    if seed is not None:
        random.seed(seed)

    greedy = solve_greedy(problem)
    current_schedule: Schedule = greedy["schedule"]
    current_objective: float = float(greedy["objective"])

    best_schedule: Schedule = copy_schedule(current_schedule)
    best_objective: float = current_objective

    # Tenure rule (transparent + instance-size dependent)
    if tabu_tenure is None:
        base = max(3, int(0.1 * len(problem.tasks)))
        tabu_tenure = base

    tabu_until: Dict[Tuple, int] = {}
    eval_cache: Dict[Tuple[Tuple[int, ...], ...], Dict] = {}

    def eval_cached(s: Schedule) -> Dict:
        key = _schedule_key(problem, s)
        if key not in eval_cache:
            eval_cache[key] = evaluate_schedule(problem, s)
        return eval_cache[key]

    for iteration in range(max_iteration):
        candidate_schedule, candidate_objective, candidate_move = local_search_step(
            problem,
            current_schedule,
            tabu_until,
            iteration,
            best_objective,
            eval_cached,
            move_types=move_types,
            max_neighbors=max_neighbors,
        )

        if candidate_schedule is None or candidate_move is None:
            break

        current_schedule = candidate_schedule
        current_objective = candidate_objective

        if current_objective < best_objective:
            best_objective = current_objective
            best_schedule = copy_schedule(current_schedule)

        # Add tabu (forbid immediate reversal)
        rev_sig = reverse_move_signature(candidate_move)
        tenure = tabu_tenure + (random.randint(0, tenure_jitter) if tenure_jitter > 0 else 0)
        tabu_until[rev_sig] = iteration + tenure

        # Cleanup expired entries (cheap)
        if iteration % 10 == 0 and tabu_until:
            expired = [m for m, until in tabu_until.items() if until <= iteration]
            for m in expired:
                tabu_until.pop(m, None)

    return best_schedule, best_objective

def copy_schedule(schedule: Schedule) -> Schedule:
    return {qc: tasks.copy() for qc, tasks in schedule.items()}

def relocate_task(
    schedule: Schedule,
    task: int,
    from_qc: int,
    to_qc: int,
    to_pos: Optional[int] = None,
) -> Schedule:
    new_schedule = copy_schedule(schedule)
    from_list = new_schedule[from_qc]
    to_list = new_schedule[to_qc]

    from_list.remove(task)
    if to_pos is None:
        to_list.append(task)
    else:
        to_list.insert(to_pos, task)
    return new_schedule

def swap_tasks(
    schedule: Schedule,
    qc_a: int,
    idx_a: int,
    qc_b: int,
    idx_b: int,
) -> Schedule:
    new_schedule = copy_schedule(schedule)
    new_schedule[qc_a][idx_a], new_schedule[qc_b][idx_b] = new_schedule[qc_b][idx_b], new_schedule[qc_a][idx_a]
    return new_schedule

def move_signature(move: Tuple) -> Tuple:
    return move

def reverse_move_signature(move: Tuple) -> Tuple:
    mtype = move[0]
    if mtype == "relocate":
        _, t, from_qc, to_qc = move
        return ("relocate", t, to_qc, from_qc)
    if mtype == "swap":
        # swap is its own reverse
        return move
    return move

def generate_neighbors(
    schedule: Schedule,
    problem,
    move_types: Sequence[str] = ("relocate", "swap"),
    max_neighbors: Optional[int] = None,
) -> List[Tuple[Schedule, Tuple]]:
    neighbors: List[Tuple[Schedule, Tuple]] = []
    qcs = list(problem.qcs)

    if "relocate" in move_types:
        for from_qc in qcs:
            for t in schedule[from_qc]:
                for to_qc in qcs:
                    if to_qc == from_qc:
                        continue
                    # insertion positions, including front
                    for pos in range(len(schedule[to_qc]) + 1):
                        new_schedule = relocate_task(schedule, t, from_qc, to_qc, to_pos=pos)
                        move = ("relocate", t, from_qc, to_qc)
                        neighbors.append((new_schedule, move))
                        if max_neighbors is not None and len(neighbors) >= max_neighbors:
                            return neighbors

    if "swap" in move_types:
        for i in range(len(qcs)):
            for j in range(i + 1, len(qcs)):
                qc_a, qc_b = qcs[i], qcs[j]
                for idx_a, ta in enumerate(schedule[qc_a]):
                    for idx_b, tb in enumerate(schedule[qc_b]):
                        new_schedule = swap_tasks(schedule, qc_a, idx_a, qc_b, idx_b)
                        # canonical signature (to avoid duplicates)
                        move = ("swap", min(ta, tb), min(qc_a, qc_b), max(ta, tb), max(qc_a, qc_b))
                        neighbors.append((new_schedule, move))
                        if max_neighbors is not None and len(neighbors) >= max_neighbors:
                            return neighbors

    return neighbors

def select_best_neighbor(
    problem,
    neighbors: Iterable[Tuple[Schedule, Tuple]],
    tabu_until: Dict[Tuple, int],
    iteration: int,
    best_objective: float,
    eval_cached,
) -> Tuple[Optional[Schedule], float, Optional[Tuple]]:
    best_obj = float("inf")
    best_schedule: Optional[Schedule] = None
    best_move: Optional[Tuple] = None

    for sched, move in neighbors:
        sig = move_signature(move)
        is_tabu = tabu_until.get(sig, -1) > iteration

        try:
            result = eval_cached(sched)
        except Exception:
            continue

        obj = float(result["objective"])
        if is_tabu and obj >= best_objective:
            continue  # tabu and no aspiration

        if obj < best_obj:
            best_obj = obj
            best_schedule = sched
            best_move = sig

    return best_schedule, best_obj, best_move

def local_search_step(
    problem,
    current_schedule: Schedule,
    tabu_until: Dict[Tuple, int],
    iteration: int,
    best_objective: float,
    eval_cached,
    move_types: Sequence[str] = ("relocate", "swap"),
    max_neighbors: Optional[int] = None,
) -> Tuple[Optional[Schedule], Optional[float], Optional[Tuple]]:
    neighbors = generate_neighbors(current_schedule, problem, move_types=move_types, max_neighbors=max_neighbors)
    best_schedule, best_obj, best_move = select_best_neighbor(
        problem,
        neighbors,
        tabu_until,
        iteration,
        best_objective,
        eval_cached,
    )
    if best_schedule is None or best_move is None or best_obj == float("inf"):
        return None, None, None
    return best_schedule, best_obj, best_move

