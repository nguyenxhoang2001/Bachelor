from greedysolver import solve_greedy,evaluate_schedule
from qc_problem import QCProblem

problem = QCProblem.generate(
    num_bays=7,
    num_qcs=3,
    total_tasks=6,
    seed=42
)

def solve_tabu(problem, max_iteration = 5):
    tabu_list = []
    tabu_tenure = 5

    greedy = solve_greedy(problem)
    current_schedule = greedy["schedule"]
    current_objective = greedy["objective"]
    best_schedule = current_schedule
    best_objective = current_objective

    for iteration in range(max_iteration):
        candidate_schedule, candidate_objective, candidate_move = local_search_step(
            problem, current_schedule, tabu_list)
        
        if candidate_schedule is None:
            break

        current_schedule = candidate_schedule
        current_objective = candidate_objective
        
        if current_objective < best_objective:
            best_objective = current_objective
            best_schedule = current_schedule

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

def generate_relocation_neighbors(schedule,problem):
    neighbors = []
    for k in problem.qcs:
        for t in schedule[k]:
                for j in problem.qcs:
                    if k != j:
                        new_schedule = relocate_task(schedule,t,k,j)
                        move = (t, k , j) #task T from qc K to qc J
                        neighbors.append((new_schedule, move))
    return neighbors

def select_best_neighbor(problem, neighbors, tabu_list):
    best_obj = float("inf")
    best_schedule = None
    best_move = None

    for neighbor in neighbors:
        schedule, move = neighbor
        if move in tabu_list:
            continue
        try:
            result = evaluate_schedule(problem, schedule)
        except:
            continue
        if result["objective"] < best_obj:
            best_obj = result["objective"]
            best_schedule = schedule
            best_move = move

    return best_schedule, best_obj, best_move

def local_search_step(problem, current_schedule, tabu_list):
    neighbors = generate_relocation_neighbors(current_schedule, problem)
    best_schedule, best_obj, best_move = select_best_neighbor(problem, neighbors, tabu_list)
    return best_schedule, best_obj, best_move

