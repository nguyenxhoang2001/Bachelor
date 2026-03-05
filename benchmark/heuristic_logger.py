"""
Heuristic solver with progress tracking (anytime trace).
"""

import time
from typing import Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qc_problem import QCProblem
from tabu_solver import solve_tabu
from greedysolver import check_interference


def solve_heuristic_with_tracking(
    problem: QCProblem,
    time_limit: float,
    max_iterations: int = 100000,
    no_improve_limit: int = None,
    seed: int = 42
) -> Tuple[Dict, List[Dict]]:
    """
    Run heuristic with progress tracking.
    
    Returns:
        (result_dict, timeseries) where:
        - result_dict: final objective, runtime, iterations, feasibility
        - timeseries: list of (time, best_objective) events
    """
    start_time = time.time()
    
    # Run tabu with trace (already returns trace)
    schedule, objective, trace = solve_tabu(
        problem,
        max_iteration=max_iterations,
        max_time=time_limit,
        no_improve_limit=no_improve_limit,
        tabu_tenure=None,
        tenure_jitter=3,
        move_types=("relocate", "swap", "intra_insert"),
        max_neighbors=None,
        seed=seed,
        verbose=False,
        return_trace=True
    )
    
    runtime = time.time() - start_time
    
    # Check feasibility
    is_feasible, violation = check_interference(problem, schedule)
    
    # Convert trace format
    timeseries = []
    for entry in trace:
        timeseries.append({
            "time": entry.get("time", 0.0),
            "best_objective": entry.get("best_objective", objective),
        })
    
    result = {
        "objective": objective if is_feasible else None,
        "runtime": runtime,
        "iterations": len(trace),
        "is_feasible": is_feasible,
        "violation": violation if not is_feasible else None,
    }
    
    return result, timeseries
