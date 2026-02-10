import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import the benchmark generator to leverage consistent instance creation.
try:
    # The generator is located in the ``share`` package.  We alias the
    # function and dataclass to avoid naming conflicts with this module.
    from share.generator import generate_instance as _generate_instance
    from share.generator import Task as _BasicTask
    HAS_BENCHMARK_GENERATOR = True
except ImportError:
    # Fallback: if the benchmark generator is not available, set flag to False.
    HAS_BENCHMARK_GENERATOR = False


@dataclass(frozen=True)
class Task:
    """Represents a single container handling task used in MILP modelling.

    Attributes
    ----------
    id : int
        Unique identifier of the task (1-based).
    bay : int
        Bay index on the vessel where the task is located.
    operation : str
        Operation type (discharge 'D' or load 'L'); retained for
        backward compatibility but not used in MILP constraints.
    level : str
        Level of the operation (e.g., 'deck' or 'hold'); retained for
        backward compatibility but not used in MILP constraints.
    duration : float
        Processing time of the task.
    """

    id: int
    bay: int
    operation: str  # 'D' or 'L'
    level: str  # 'deck' or 'hold'
    duration: float


class QCProblem:
    def __init__(self):
        """Initialize an empty QCProblem instance."""
        pass

    @classmethod
    def generate(
        cls,
        num_bays: int,
        num_qcs: int,
        total_tasks: int,
        processing_time_range: Tuple[float, float] = (3.0, 180.0),
        travel_per_bay: float = 1.0,
        seed: Optional[int] = None,
    ):
        """Generate a QCProblem instance.

        This method leverages the benchmark generator if available to create
        tasks with precedence relations and converts them into the data
        structures expected by the MILP model.  If the benchmark generator
        is not available, it falls back to a simple random generation similar
        to the original implementation.

        Parameters
        ----------
        num_bays : int
            Number of bays on the vessel.
        num_qcs : int
            Number of quay cranes.
        total_tasks : int
            Number of container handling tasks to generate.
        processing_time_range : Tuple[float, float], optional
            Range of processing times (ignored when using the benchmark generator,
            which internally defines its own range).  Retained for backward
            compatibility.
        travel_per_bay : float, optional
            Travel time per bay for cranes.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        QCProblem
            The generated problem instance.
        """
        if seed is not None:
            random.seed(seed)

        prob = cls()
        prob.qcs = list(range(1, num_qcs + 1))

        # Use benchmark generator if available
        if HAS_BENCHMARK_GENERATOR:
            # Generate an instance using the shared generator.  We force
            # allow_precedence=True to create precedence chains.
            instance = _generate_instance(
                n_tasks=total_tasks,
                n_cranes=num_qcs,
                max_bay=num_bays,
                seed=seed,
                allow_precedence=True,
            )
            # Convert tasks to local Task dataclass and reassign IDs starting from 1
            tasks: List[Task] = []
            id_map: Dict[int, int] = {}
            for idx, t in enumerate(instance.tasks):
                new_id = idx + 1
                id_map[t.id] = new_id
                tasks.append(
                    Task(
                        id=new_id,
                        bay=t.bay,
                        operation='D',  # default placeholder
                        level='deck',  # default placeholder
                        duration=float(t.processing_time),
                    )
                )
            prob.tasks = [t.id for t in tasks]
            # Parameters
            prob.duration = {t.id: t.duration for t in tasks}
            prob.location = {t.id: t.bay for t in tasks}
            prob.operation = {t.id: t.operation for t in tasks}
            prob.level = {t.id: t.level for t in tasks}
            # Precedence Φ from instance predecessors
            phi: List[Tuple[int, int]] = []
            for t in instance.tasks:
                for pred in t.predecessors:
                    # Map original IDs to new IDs
                    phi.append((id_map[pred], id_map[t.id]))
            prob.phi = phi
            # Non-simultaneity Ψ: tasks within one bay distance
            psi: List[Tuple[int, int]] = []
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    if abs(tasks[i].bay - tasks[j].bay) <= 1:
                        psi.append((tasks[i].id, tasks[j].id))
            prob.psi = psi
            # QC travel times based on travel_per_bay.  Use travel_per_bay if provided,
            # otherwise derive from instance.move_time.
            move_time = getattr(instance, 'move_time', travel_per_bay)
            prob.earliest_time = {k: 0.0 for k in prob.qcs}
            prob.starting_travel_time = {}
            prob.travel_time = {}
            prob.final_travel_time = {}
            for k in prob.qcs:
                start_bay = (
                    int((k - 1) * (num_bays - 1) / (num_qcs - 1)) + 1
                    if num_qcs > 1 else 1
                )
                for t in tasks:
                    prob.starting_travel_time[(k, t.id)] = move_time * abs(start_bay - t.bay)
                for ti in tasks:
                    for tj in tasks:
                        prob.travel_time[(k, ti.id, tj.id)] = move_time * abs(ti.bay - tj.bay)
                for t in tasks:
                    prob.final_travel_time[(k, t.id)] = 0.0
            # Constants
            prob.M = 10000
            prob.alpha1 = 100
            prob.alpha2 = 1
            return prob
        else:
            # Fallback to original random generation logic if benchmark generator
            # is not available.
            tasks: List[Task] = []
            for tid in range(1, total_tasks + 1):
                bay = random.randint(1, num_bays)
                operation = random.choice(['D', 'L'])
                level = random.choice(['deck', 'hold'])
                duration = random.uniform(*processing_time_range)
                tasks.append(
                    Task(
                        id=tid,
                        bay=bay,
                        operation=operation,
                        level=level,
                        duration=duration,
                    )
                )
            # Sort and reassign IDs for solver stability
            tasks.sort(key=lambda t: (t.bay, t.operation, t.level))
            id_map = {old.id: i + 1 for i, old in enumerate(tasks)}
            tasks = [
                Task(
                    id=id_map[t.id],
                    bay=t.bay,
                    operation=t.operation,
                    level=t.level,
                    duration=t.duration,
                )
                for t in tasks
            ]
            prob.tasks = [t.id for t in tasks]
            # Set parameter dictionaries
            prob.duration = {t.id: t.duration for t in tasks}
            prob.location = {t.id: t.bay for t in tasks}
            prob.operation = {t.id: t.operation for t in tasks}
            prob.level = {t.id: t.level for t in tasks}
            # Build precedence relations
            prob.phi = []
            by_bay: Dict[int, List[Task]] = {}
            for t in tasks:
                by_bay.setdefault(t.bay, []).append(t)
            for bay, bay_tasks in by_bay.items():
                discharge = [t for t in bay_tasks if t.operation == 'D']
                load = [t for t in bay_tasks if t.operation == 'L']
                for d in discharge:
                    for l in load:
                        prob.phi.append((d.id, l.id))
                discharge_deck = [t for t in discharge if t.level == 'deck']
                discharge_hold = [t for t in discharge if t.level == 'hold']
                for dd in discharge_deck:
                    for dh in discharge_hold:
                        prob.phi.append((dd.id, dh.id))
                load_hold = [t for t in load if t.level == 'hold']
                load_deck = [t for t in load if t.level == 'deck']
                for lh in load_hold:
                    for ld in load_deck:
                        prob.phi.append((lh.id, ld.id))
            # Non-simultaneity relations
            prob.psi = []
            for i in range(len(tasks)):
                for j in range(i + 1, len(tasks)):
                    if abs(tasks[i].bay - tasks[j].bay) <= 1:
                        prob.psi.append((tasks[i].id, tasks[j].id))
            # Travel times
            prob.earliest_time = {k: 0.0 for k in prob.qcs}
            prob.starting_travel_time = {}
            prob.travel_time = {}
            prob.final_travel_time = {}
            for k in prob.qcs:
                start_bay = (
                    int((k - 1) * (num_bays - 1) / (num_qcs - 1)) + 1
                    if num_qcs > 1 else 1
                )
                for t in tasks:
                    prob.starting_travel_time[(k, t.id)] = travel_per_bay * abs(start_bay - t.bay)
                for ti in tasks:
                    for tj in tasks:
                        prob.travel_time[(k, ti.id, tj.id)] = travel_per_bay * abs(ti.bay - tj.bay)
                for t in tasks:
                    prob.final_travel_time[(k, t.id)] = 0.0
            prob.M = 10000
            prob.alpha1 = 100
            prob.alpha2 = 1
            return prob
