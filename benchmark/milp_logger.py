"""
MILP solver with Gurobi callback for progress tracking.
"""

import time
from typing import Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from milp_solver import solve_milp
from qc_problem import QCProblem


def solve_milp_with_tracking(
    problem: QCProblem,
    time_limit: float,
    threads: int = 1
) -> Tuple[Dict, List[Dict]]:
    """
    Solve MILP with log parsing for progress tracking.
    
    Returns:
        (result_dict, timeseries) where:
        - result_dict: final status, objective, bound, runtime, gap
        - timeseries: list of progress events parsed from log
    """
    import tempfile
    import os
    
    # Create temporary log file
    log_fd, log_file = tempfile.mkstemp(suffix='.log', text=True)
    os.close(log_fd)
    
    try:
        start_time = time.time()
        
        # Run MILP with logging
        result = solve_milp(
            problem,
            time_limit=time_limit,
            verbose=False,
            log_file=log_file
        )
        
        runtime = time.time() - start_time
        
        # Parse log for time series
        timeseries = parse_gurobi_log_timeseries(log_file, start_time)
        
        # Format result
        result_dict = {
            "status": result.get("status", "UNKNOWN"),
            "objective": result.get("objective"),
            "best_bound": result.get("best_bound"),
            "gap": result.get("gap"),
            "runtime": runtime,
            "node_count": result.get("node_count", 0),
            "sol_count": result.get("sol_count", 0),
        }
        
        return result_dict, timeseries
        
    finally:
        # Clean up temp file
        try:
            os.remove(log_file)
        except:
            pass


def parse_gurobi_log_timeseries(log_file: str, start_time: float) -> List[Dict]:
    """
    Parse Gurobi log to extract incumbent/bound trajectory.
    
    Returns:
        List of dicts with keys: time, incumbent, best_bound, gap, event
    """
    import re
    
    timeseries = []
    
    if not os.path.exists(log_file):
        return timeseries
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        # Initial solution: "Found heuristic solution: objective ..."
        if "Found heuristic solution: objective" in line:
            match = re.search(r'objective\s+([\d.e+-]+)', line)
            if match:
                obj = float(match.group(1))
                timeseries.append({
                    "time": 0.0,
                    "incumbent": obj,
                    "best_bound": None,
                    "gap": None,
                    "event": "initial"
                })
        
        # Node log lines with incumbent improvements (marked with 'H' or '*')
        if line.strip().startswith('H') or line.strip().startswith('*'):
            parts = line.split()
            if len(parts) >= 7:
                try:
                    # Extract time (last column with 's' suffix)
                    time_str = parts[-1].replace('s', '')
                    t = float(time_str)
                    
                    # Find incumbent and best bound
                    # Format: H/* [nodes] [cuts] [incumbent] [bestbd] [gap%] [it/n] [time]
                    for i, part in enumerate(parts):
                        if '%' in part:
                            # Gap column found, incumbent and bound are before it
                            if i >= 2:
                                try:
                                    incumbent = float(parts[i - 2])
                                    best_bound = float(parts[i - 1])
                                    gap_str = part.replace('%', '')
                                    gap = float(gap_str) if gap_str else None
                                    
                                    timeseries.append({
                                        "time": t,
                                        "incumbent": incumbent,
                                        "best_bound": best_bound,
                                        "gap": gap,
                                        "event": "improvement"
                                    })
                                    break
                                except ValueError:
                                    continue
                except (ValueError, IndexError):
                    continue
    
    return timeseries
