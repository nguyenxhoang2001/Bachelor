import math
from greedysolver import solve_greedy, evaluate_schedule
from qc_problem import QCProblem


def solve_tabu(problem, max_iteration=100):
    tabu_list = []
    # Dynamic tabu tenure based on problem size (well-established rule of thumb)
    tabu_tenure = max(5, int(math.sqrt(len(problem.tasks))))

    greedy = solve_greedy(problem)
    current_schedule = greedy["schedule"]
    current_objective = greedy["objective"]
    best_schedule = current_schedule
    best_objective = current_objective

    for iteration in range(max_iteration):
        candidate_schedule, candidate_objective, candidate_move = local_search_step(
            problem, current_schedule, tabu_list, best_objective)

        if candidate_schedule is None:
            break

        current_schedule = candidate_schedule
        current_objective = candidate_objective

        if current_objective < best_objective:
            best_objective = current_objective
            best_schedule = current_schedule

        # Add the reverse move to the tabu list to prevent cycling
        if candidate_move[0] == 'swap':
            _, t1, k1, t2, k2 = candidate_move
            tabu_list.append(('swap', t2, k2, t1, k1))
        else:
            t, from_qc, to_qc = candidate_move
            tabu_list.append((t, to_qc, from_qc))

        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return best_schedule, best_objective


def copy_schedule(schedule):
    new_schedule = {}
    for qc in schedule:
        new_schedule[qc] = schedule[qc].copy()
    return new_schedule


def relocate_task(schedule, task, from_qc, to_qc):
    new_schedule = copy_schedule(schedule)
    new_schedule[from_qc].remove(task)
    new_schedule[to_qc].append(task)
    return new_schedule


def swap_tasks(schedule, t1, k1, t2, k2):
    """Swap task t1 (assigned to k1) with task t2 (assigned to k2)."""
    new_schedule = copy_schedule(schedule)
    new_schedule[k1].remove(t1)
    new_schedule[k1].append(t2)
    new_schedule[k2].remove(t2)
    new_schedule[k2].append(t1)
    return new_schedule


def generate_relocation_neighbors(schedule, problem):
    neighbors = []
    for k in problem.qcs:
        for t in schedule[k]:
            for j in problem.qcs:
                if k != j:
                    new_schedule = relocate_task(schedule, t, k, j)
                    move = (t, k, j)  # task T from qc K to qc J
                    neighbors.append((new_schedule, move))
    return neighbors


def generate_swap_neighbors(schedule, problem):
    """Generate neighbors by swapping one task from QC k1 with one task from QC k2."""
    neighbors = []
    qcs = problem.qcs
    for i, k1 in enumerate(qcs):
        for k2 in qcs[i + 1:]:
            for t1 in schedule[k1]:
                for t2 in schedule[k2]:
                    new_schedule = swap_tasks(schedule, t1, k1, t2, k2)
                    move = ('swap', t1, k1, t2, k2)
                    neighbors.append((new_schedule, move))
    return neighbors


def select_best_neighbor(problem, neighbors, tabu_list, best_obj_so_far):
    best_obj = float("inf")
    best_schedule = None
    best_move = None

    for neighbor in neighbors:
        schedule, move = neighbor
        is_tabu = move in tabu_list
        try:
            result = evaluate_schedule(problem, schedule)
        except Exception:
            continue
        obj = result["objective"]
        # Aspiration criterion: allow a tabu move if it improves the best known solution
        if is_tabu and obj >= best_obj_so_far:
            continue
        if obj < best_obj:
            best_obj = obj
            best_schedule = schedule
            best_move = move

    return best_schedule, best_obj, best_move


def local_search_step(problem, current_schedule, tabu_list, best_obj_so_far):
    neighbors = (
        generate_relocation_neighbors(current_schedule, problem) +
        generate_swap_neighbors(current_schedule, problem)
    )
    best_schedule, best_obj, best_move = select_best_neighbor(
        problem, neighbors, tabu_list, best_obj_so_far)
    return best_schedule, best_obj, best_move


if __name__ == "__main__":
    problem = QCProblem.generate(
        num_bays=7,
        num_qcs=3,
        total_tasks=6,
        seed=42
    )
    best_schedule, best_obj = solve_tabu(problem)
    print(f"Best objective: {best_obj}")
    for qc, tasks in best_schedule.items():
        print(f"  QC {qc}: {tasks}")

