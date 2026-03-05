# QCSP Benchmark Framework

Reproducible benchmark framework to compare MILP (Gurobi) solver against Greedy+Tabu heuristic for the Quay Crane Scheduling Problem (QCSP).

## Overview

This framework runs two benchmark suites:

- **Part A**: Optimality Check Suite (small instances where MILP proves optimality)
- **Part B**: Primal Bound Race Suite (larger instances with time-series comparison)

## Installation

### Prerequisites

- Python 3.7+
- Gurobi solver with valid license
- Required packages:
  ```bash
  pip install numpy matplotlib pyyaml
  ```

### Setup

Ensure your project has the following structure:
```
Bachelor/
├── qc_problem.py          # QCSP problem definition
├── milp_solver.py         # MILP solver wrapper
├── tabu_solver.py         # Tabu search implementation
├── greedysolver.py        # Greedy constructor
├── benchmark/             # Benchmark framework (this package)
│   ├── __init__.py
│   ├── common.py
│   ├── instance_factory.py
│   ├── milp_logger.py
│   ├── heuristic_logger.py
│   ├── run_partA.py
│   ├── run_partB.py
│   ├── plot_results.py
│   └── cli.py
└── configs/
    └── benchmark.yaml     # Configuration file
```

## Quick Start

### 1. Run Benchmarks

```bash
# Run Part A (optimality check on small instances)
python -m benchmark.cli run --config configs/benchmark.yaml --suite partA --output results/

# Run Part B (primal bound race on larger instances)
python -m benchmark.cli run --config configs/benchmark.yaml --suite partB --output results/

# Run both suites
python -m benchmark.cli run --config configs/benchmark.yaml --suite both --output results/
```

### 2. Generate Plots

```bash
# Generate plots for both suites
python -m benchmark.cli plot --input results/ --suite both
```

## Configuration

Edit `configs/benchmark.yaml` to customize:

- **Instance parameters**: task range, bays, precedence probability, processing times
- **MILP settings**: time limit, threads
- **Heuristic settings**: time limit, iterations, seed
- **Part A**: task sizes (5-9), QC counts, instances per config
- **Part B**: task sizes (10-20), QC counts, time checkpoints

## Output Files

### Part A (Optimality Check)

- **partA_results.csv**: Detailed results for each instance
  - Columns: instance_id, config_id, num_tasks, num_qcs, seed, MILP/heuristic objectives, runtimes, optimality match, gap to optimal
  
- **partA_summary.csv**: Aggregated statistics by configuration
  - Columns: config_id, optimal_match_rate, mean_gap, mean_runtimes

- **Figures**:
  - `partA_optimality_match_rate.png`: Bar chart of optimality match rate by size
  - `partA_gap_distribution.png`: Box plot of gap distribution
  - `partA_runtime_comparison.png`: Scatter plot of MILP vs heuristic runtime

### Part B (Primal Bound Race)

- **partB_runs.csv**: Summary results for each instance
  - Columns: instance_id, config_id, MILP/heuristic final objectives, runtimes, winner, checkpoint objectives

- **partB_timeseries.csv**: Time-series data (long format)
  - Columns: instance_id, method, time, objective, best_bound, gap

- **partB_summary.csv**: Aggregated statistics by configuration
  - Columns: config_id, MILP/heuristic wins, mean_runtimes, mean_gap

- **Figures**:
  - `partB_winner_distribution.png`: Bar chart of winner counts by config
  - `partB_checkpoint_comparison.png`: Average objective at time checkpoints
  - `partB_timeseries_instX.png`: Time-series plots for representative instances

## Metrics Explanation

### Part A Metrics

- **Optimality Match Rate**: Percentage of instances where heuristic finds optimal solution
- **Gap to Optimal**: (heuristic_obj - optimal_obj) / optimal_obj * 100
- **Runtime**: Wall-clock time to completion

### Part B Metrics

- **Winner**: Which method achieved better final objective (MILP/HEURISTIC/TIE)
- **Checkpoints**: Best objective achieved by each method at specific time points
- **Gap MILP-Heur**: (heuristic_obj - milp_obj) / heuristic_obj * 100

## Research Question Addressed

**How do heuristic methods compare to an exact formulation in terms of solution quality, computational efficiency, and scalability for the QCSP?**

### Solution Quality
- Part A: Optimality match rate shows how often heuristic finds provably optimal solutions
- Part B: Winner distribution and gap metrics show relative solution quality on harder instances
- Part B: Lower bound comparison (MILP best_bound) provides quality certificates

### Computational Efficiency
- Runtime comparison at equal time limits
- Time-to-target analysis: which method reaches good solutions faster
- Part B time-series: anytime performance comparison

### Scalability
- Part A: Small instances (5-9 tasks) where MILP proves optimality quickly
- Part B: Larger instances (10-20 tasks) where MILP struggles
- Performance degradation as instance size increases

## Advanced Usage

### Custom Instance Sizes

Edit `configs/benchmark.yaml`:
```yaml
part_a:
  task_sizes: [3, 4, 5, 6, 7, 8, 9]  # Extend range
  qc_counts: [2, 3, 4]                # Add more QCs

part_b:
  task_sizes: [10, 15, 20, 25, 30]   # Test larger sizes
  checkpoints: [10, 30, 60, 300, 600] # Custom time points
```

### Multiple Heuristic Runs

To test heuristic variability, run multiple seeds:
```python
# In heuristic_logger.py, modify solve_heuristic_with_tracking()
# to accept and vary the seed parameter
```

### Warm-Start MILP from Heuristic

Modify `milp_solver.py` to accept initial solution:
```python
# Use model.start or variable.Start attributes
# Feed heuristic solution to MILP solver
```

## Troubleshooting

### ImportError: No module named 'yaml'
```bash
pip install pyyaml
```

### Gurobi License Error
Ensure Gurobi is properly installed and licensed:
```bash
gurobi_cl --license
```

### Memory Issues on Large Instances
Reduce `num_instances_per_config` or `task_sizes` in config file.

## Citation

If you use this benchmark framework, please cite your research paper and acknowledge the use of Gurobi solver.

## License

[Your License Here]
