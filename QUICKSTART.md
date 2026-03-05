# Quick Start Guide - QCSP Benchmark Framework

## Test the Installation

First, test that everything works:

```bash
# From the Bachelor directory
python test_benchmark.py
```

This will run a small integration test with 5 tasks to verify:
- Instance generation works
- Heuristic tracking works
- MILP tracking works
- Results comparison works

## Run Your First Benchmark

### Step 1: Configure the Benchmark

Edit `configs/benchmark.yaml` if needed. The default configuration is:
- Part A: 5-9 tasks, 2-3 QCs, 20 instances per config
- Part B: 10-20 tasks, 2-3 QCs, 10 instances per config
- Time limit: 1000 seconds for both MILP and heuristic

### Step 2: Run Part A (Small Instances)

```bash
python -m benchmark.cli run --config configs/benchmark.yaml --suite partA --output results/
```

Expected runtime: ~30-60 minutes (depending on your machine)

Output files in `results/`:
- `partA_results.csv` - detailed per-instance results
- `partA_summary.csv` - aggregated statistics

### Step 3: Generate Part A Plots

```bash
python -m benchmark.cli plot --input results/ --suite partA
```

Plots will be saved to `results/figures/`:
- `partA_optimality_match_rate.png`
- `partA_gap_distribution.png`
- `partA_runtime_comparison.png`

### Step 4: Run Part B (Larger Instances)

```bash
python -m benchmark.cli run --config configs/benchmark.yaml --suite partB --output results/
```

Expected runtime: ~2-4 hours (depending on configuration)

Output files in `results/`:
- `partB_runs.csv` - summary per instance
- `partB_timeseries.csv` - time-series data
- `partB_summary.csv` - aggregated statistics

### Step 5: Generate Part B Plots

```bash
python -m benchmark.cli plot --input results/ --suite partB
```

Plots will be saved to `results/figures/`:
- `partB_winner_distribution.png`
- `partB_checkpoint_comparison.png`
- `partB_timeseries_instX.png` (multiple files for representative instances)

## Run Everything at Once

```bash
# Run both benchmarks
python -m benchmark.cli run --config configs/benchmark.yaml --suite both --output results/

# Generate all plots
python -m benchmark.cli plot --input results/ --suite both
```

## Customize for Your Research

### Quick Benchmark (Testing)

Edit `configs/benchmark.yaml`:
```yaml
part_a:
  task_sizes: [5, 6, 7]  # Fewer sizes
  qc_counts: [2]          # Fewer QCs
  num_instances_per_config: 5  # Fewer instances

part_b:
  task_sizes: [10, 12]    # Fewer sizes
  qc_counts: [2]
  num_instances_per_config: 3

milp:
  time_limit: 300.0  # Shorter time limit

heuristic:
  time_limit: 300.0
```

### Extended Benchmark (Publication Quality)

```yaml
part_a:
  task_sizes: [3, 4, 5, 6, 7, 8, 9]
  qc_counts: [2, 3, 4]
  num_instances_per_config: 30  # More instances for statistical significance

part_b:
  task_sizes: [10, 12, 15, 18, 20, 25, 30]
  qc_counts: [2, 3, 4]
  num_instances_per_config: 20
  checkpoints: [1, 5, 10, 30, 60, 120, 300, 600, 1000, 1800]

milp:
  time_limit: 1800.0  # Longer time limit

heuristic:
  time_limit: 1800.0
```

## Interpreting Results

### Part A Results

**partA_summary.csv** shows:
- `optimal_match_rate_percent`: How often heuristic finds optimal solution
- `mean_gap_to_optimal_percent`: Average gap when not optimal
- `mean_milp_runtime` vs `mean_heuristic_runtime`: Efficiency comparison

**Key Questions:**
- Does heuristic reach optimal on small instances?
- How does match rate degrade with instance size?
- Is heuristic faster even when reaching optimal?

### Part B Results

**partB_summary.csv** shows:
- `milp_wins` vs `heuristic_wins`: Solution quality comparison
- `mean_gap_percent`: Average quality difference
- Runtimes: Computational efficiency

**Key Questions:**
- Which method finds better solutions on hard instances?
- How does performance change with instance size?
- Time-to-target: Does heuristic find good solutions faster?

## Troubleshooting

### "ModuleNotFoundError: No module named 'yaml'"
```bash
pip install pyyaml
```

### "Gurobi license error"
Make sure Gurobi is installed and licensed:
```bash
gurobi_cl --license
```

### "No such file or directory: configs/benchmark.yaml"
Make sure you're running from the Bachelor directory and the config file exists.

### Very long runtime
Reduce instance counts or time limits in the config file for testing.

### Out of memory
Reduce `num_instances_per_config` or use smaller `task_sizes`.

## Next Steps

After running the benchmarks:

1. **Analyze the CSVs**: Import into Excel/Python for detailed analysis
2. **Review the plots**: Visualizations suitable for presentations/papers
3. **Statistical tests**: Run paired tests on the results (Wilcoxon, t-test)
4. **Write up**: Use the metrics to answer your research question

## Support

For issues or questions:
1. Check the full README: `benchmark/README.md`
2. Review the code documentation in each module
3. Run the integration test: `python test_benchmark.py`
