import gurobipy as gp
from gurobipy import GRB
from qc_problem import QCProblem


def build_model(problem):
    model = gp.Model("QC Schedule Model")

    X = {} # 1, if QC k performs task j after task i; 0, otherwise. 
    for k in problem.qcs:

        for j in problem.tasks: #Every first task for k * qcs
            X[(k,0,j)] = model.addVar(vtype=GRB.BINARY, name = f"X_{k}_0_{j}")
    
        for j in problem.tasks: #Every ending task for for k * qcs
            X[(k,j,'T')] = model.addVar(vtype=GRB.BINARY, name = f"X_{k}_{j}_T")

        for i in problem.tasks:  #Every other combination
            for j in problem.tasks:
                if i != j:
                    X[(k,i,j)] = model.addVar(vtype=GRB.BINARY, name = f"X_{k}_{i}_{j}")

    
    Y = {} # Completion time of QC_k
    for k in problem.qcs:
        Y[k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name = f"Y_{k}")

    D = {} # Completion time of task i
    for i in problem.tasks:
        D[i] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name = f"D_{i}")

    Z = {} # 1, if task j starts later than the completion time of task i; 0, otherwise.
    for i in problem.tasks:
        for j in problem.tasks:
            if i != j:
                Z[(i,j)] = model.addVar(vtype=GRB.BINARY, name = f"Z_{i}_{j}")

    W = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name = f"W")

    return model, X, Y, D, Z, W 

def add_constraints(model, problem, X, Y, D, Z, W):

    for k in problem.qcs: # (2) Completion time of every QC must be smaller than the makespan
        model.addConstr(
            Y[k] <= W, name=f"makespan_{k}"
            )

    for k in problem.qcs: # (3) Starting task for every QC
        model.addConstr(
            gp.quicksum(X[(k, 0, j)] for j in problem.tasks) == 1, name=f"start_{k}"
            )

    for k in problem.qcs: # (4) Final task for every QC
        model.addConstr(
            gp.quicksum(X[(k, i, 'T')] for i in problem.tasks) == 1, name=f"end_{k}"
            )

    for j in problem.tasks: # (5) every task completed by exactly one QC
        sum_expr = gp.LinExpr()
        for k in problem.qcs:
            for i in [0] + problem.tasks:
                if i == j:
                    continue
                if (k,i,j) in X:
                    sum_expr += X[(k,i,j)]
        model.addConstr(sum_expr == 1, name=f"task_assigned_{j}")



    for k in problem.qcs: # (6) Flow balance
        for i in problem.tasks:
            inflow = gp.LinExpr()
            outflow = gp.LinExpr()

            #inflow to task i
            for h in [0] + problem.tasks:
                if h == i:
                    continue
                if (k,h,i) in X:
                    inflow += X[(k,h,i)]

            #outflow from task i
            for j in problem.tasks + ['T']:
                if j == i:
                    continue
                if (k,i,j) in X:
                    outflow += X[(k,i,j)]
            
            model.addConstr(inflow == outflow, name = f"flow_{k}_{i}")

    for k in problem.qcs: # (7) Completion time
        for i in [0] + problem.tasks:
            for j in problem.tasks:
                if i == j or (k,i,j) not in X:
                    continue
                if i == 0: # (13) Earliest start of each QC
                    model.addConstr(
                        problem.earliest_time[k] + problem.starting_travel_time.get((k,j),0) 
                        + problem.duration[j] - D[j] <= problem.M * (1 - X[(k,0,j)]), name=f"comp_time_start_{k}_{j}"
                    )
                else:
                    model.addConstr(
                        D[i] + problem.travel_time.get((k,i, j), 0) + problem.duration[j] - D[j] 
                        <= problem.M * (1 - X[(k, i, j)]),
                        name=f"comp_time_{k}_{i}_{j}"
                    )

    for (i,j) in problem.phi: # (8) Task order
            model.addConstr(
                D[i] + problem.duration[j] <= D[j], name = f"task_order_{i}_{j}"
            )
    
    for i in problem.tasks: # (9) Task order
        for j in problem.tasks:
            if i != j:
                model.addConstr(
                    D[i] - D[j] + problem.duration[j] <= problem.M * (1 - Z[(i,j)]),
                    name=f"Z_{i}_{j}" 
                )

    for (i,j) in problem.psi: # (10) Task order pt.2
        model.addConstr(
            Z[(i,j)] + Z[(j,i)] == 1, name= f"Z_constraint_{i}_{j}"
            )
        
# (11) QC interference - simplified version
    if len(problem.qcs) > 1:
        sorted_qcs = sorted(problem.qcs)
    
        for i in problem.tasks:
            for j in problem.tasks:
                if i != j and problem.location[i] < problem.location[j]:
                    for qc_left in sorted_qcs:
                        for qc_right in sorted_qcs:
                            if qc_right > qc_left:  # right crane is to the right
                            # Check if left crane does right task AND right crane does left task
                                cond1 = gp.LinExpr()
                                cond2 = gp.LinExpr()
                            
                            # Left crane qc_left does task j (right task)
                                for prev in [0] + problem.tasks:
                                    if prev != j and (qc_left, prev, j) in X:
                                        cond1 += X[(qc_left, prev, j)]
                            
                            # Right crane qc_right does task i (left task)
                                for prev in [0] + problem.tasks:
                                    if prev != i and (qc_right, prev, i) in X:
                                        cond2 += X[(qc_right, prev, i)]
                            
                            # If both conditions hold (cond1 = 1 and cond2 = 1)
                            # Then force Z[i,j] + Z[j,i] >= 1
                                model.addConstr(
                                    cond1 + cond2 <= 1 + (Z[(i, j)] + Z[(j, i)]),
                                    name=f"interference_{i}_{j}_{qc_left}_{qc_right}"
                                )
                                        
    """
    sorted_qcs = sorted(problem.qcs)

    for i in problem.tasks: # (11) QC interference 
        for j in problem.tasks:
            if i != j and problem.location[i] < problem.location[j]:
                for k_val in range(1, len(sorted_qcs) + 1):
                    first_k_qcs = sorted_qcs[:k_val]
                    sum_j = gp.LinExpr()
                    sum_i = gp.LinExpr()
                
                    for qc in first_k_qcs: #Check if QC qc does task j
                        for prev in [0] + problem.tasks:
                            if prev != j and (qc, prev, j) in X:
                                sum_j += X[(qc, prev, j)]
                    
                        for prev in [0] + problem.tasks: #Check if QC qc does task i
                            if prev != i and (qc, prev, i) in X:
                                sum_i += X[(qc, prev, i)]
                
                    model.addConstr(
                        sum_j - sum_i <= problem.M * (Z[(i, j)] + Z[(j, i)]),
                        name=f"interference_{i}_{j}_{k_val}"
                    )
    """    
    for k in problem.qcs: # (12) Completion time of each QC
        for j in problem.tasks:
            if (k,j,'T') in X:
                model.addConstr(
                    D[j] + problem.final_travel_time.get((k,j),0) - Y[k] <= problem.M * (1 - X[(k,j,'T')]), name=f"completion_time_{k}_{j}"
                )
   
def set_objective(model, problem, Y, W):
    model.setObjective(problem.alpha1 * W + problem.alpha2 * gp.quicksum(Y[k] for k in problem.qcs), GRB.MINIMIZE)
