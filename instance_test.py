from qc_problem import QCProblem

# Generate a sample instance
instance = QCProblem.generate(
    num_bays=30,
    num_qcs=2,
    total_tasks=10,
    seed=42
)

# Display instance details
print("=" * 60)
print("GENERATED QC PROBLEM INSTANCE")
print("=" * 60)

print("\n TASKS:")
for task_id in instance.tasks:
    print(f"  Task {task_id}: Bay {instance.location[task_id]}, Duration {instance.duration[task_id]:.2f}")

print(f"\n QCs: {instance.qcs}")
print(f"Initial positions: {instance.initial_position}")
print(f"Final positions: {instance.final_position}")

print(f"\n PRECEDENCE CONSTRAINTS (Φ): {len(instance.phi)} constraints")
for (i, j) in instance.phi:
    print(f"  Task {i} → Task {j}")

print(f"\n NON-SIMULTANEITY CONSTRAINTS (Ψ): {len(instance.psi)} constraints")
for (i, j) in instance.psi[:5]:  # Show first 5
    print(f"  ({i}, {j})")
if len(instance.psi) > 5:
    print(f"  ... and {len(instance.psi) - 5} more")

print(f"\n BIG-M: {instance.M:.2f}")
print(f"Weights: α1={instance.alpha1}, α2={instance.alpha2}")