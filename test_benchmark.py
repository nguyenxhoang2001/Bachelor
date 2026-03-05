"""
Quick integration test for benchmark framework.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from qc_problem import QCProblem
from benchmark.instance_factory import InstanceFactory
from benchmark.milp_logger import solve_milp_with_tracking
from benchmark.heuristic_logger import solve_heuristic_with_tracking

def test_basic_integration():
    """Test basic integration of benchmark components."""
    
    print("Testing Benchmark Framework Integration")
    print("=" * 60)
    
    # Create instance factory
    config = {
        "processing_time_range": [3.0, 180.0],
        "travel_per_bay": 1.0,
        "precedence_probability": 0.3,
        "num_bays_mode": "fixed",
        "fixed_bays": 30
    }
    
    factory = InstanceFactory(config)
    
    # Generate small test instance
    print("\n1. Generating test instance...")
    problem, metadata = factory.generate(num_tasks=5, num_qcs=2, seed=42)
    print(f"   Generated: {metadata['num_tasks']} tasks, {metadata['num_qcs']} QCs")
    
    # Test heuristic with tracking
    print("\n2. Testing heuristic with tracking...")
    heur_result, heur_trace = solve_heuristic_with_tracking(
        problem,
        time_limit=10.0,
        max_iterations=1000,
        seed=42
    )
    print(f"   Heuristic: obj={heur_result['objective']}, "
          f"runtime={heur_result['runtime']:.2f}s, "
          f"iterations={heur_result['iterations']}, "
          f"feasible={heur_result['is_feasible']}")
    print(f"   Trace entries: {len(heur_trace)}")
    
    # Test MILP with tracking
    print("\n3. Testing MILP with tracking...")
    milp_result, milp_trace = solve_milp_with_tracking(
        problem,
        time_limit=20.0,
        threads=1
    )
    print(f"   MILP: status={milp_result['status']}, "
          f"obj={milp_result['objective']}, "
          f"runtime={milp_result['runtime']:.2f}s")
    print(f"   Trace entries: {len(milp_trace)}")
    
    # Compare results
    print("\n4. Comparison:")
    if milp_result['objective'] and heur_result['objective'] and heur_result['is_feasible']:
        gap = abs(milp_result['objective'] - heur_result['objective']) / \
              max(abs(milp_result['objective']), 1e-6) * 100
        print(f"   Gap: {gap:.2f}%")
        
        if milp_result['status'] == 'OPTIMAL':
            is_optimal = gap < 1e-6
            print(f"   Heuristic reached optimal: {is_optimal}")
    
    print("\n" + "=" * 60)
    print("Integration test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_basic_integration()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
