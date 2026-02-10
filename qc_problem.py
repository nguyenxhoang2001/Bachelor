import random
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: int
    bay: int
    operation: str      # 'D' or 'L'
    level: str          # 'deck' or 'hold'
    duration: float     # p_i


class QCProblem:
    def __init__(self):
        pass

    @classmethod
    def generate(
        cls,
        num_bays: int,
        num_qcs: int,
        total_tasks: int,
        processing_time_range: Tuple[float, float] = (3.0, 180.0),
        travel_per_bay: float = 1.0,
        seed: int = None
    ):
        if seed is not None:
            random.seed(seed)

        prob = cls()

        # ------------------
        # Sets
        # ------------------
        prob.qcs = list(range(1, num_qcs + 1))
        tasks: List[Task] = []

        # ------------------
        # Generate tasks
        # ------------------
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
                    duration=duration
                )
            )

        # Sort by bay for solver stability
        tasks.sort(key=lambda t: (t.bay, t.operation, t.level))


        # Reassign IDs
        id_map = {old.id: i + 1 for i, old in enumerate(tasks)}

        tasks = [
            Task(
                id=id_map[t.id],
                bay=t.bay,
                operation=t.operation,
                level=t.level,
                duration=t.duration
            )
            for t in tasks
        ]

        prob.tasks = [t.id for t in tasks]

        # ------------------
        # Parameters
        # ------------------
        prob.duration = {t.id: t.duration for t in tasks}
        prob.location = {t.id: t.bay for t in tasks}
        prob.operation = {t.id: t.operation for t in tasks}
        prob.level = {t.id: t.level for t in tasks}


        # ------------------
        # Precedence Φ
        # ------------------
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

    # Rule 2: Discharge - deck before hold
        discharge_deck = [t for t in discharge if t.level == 'deck']
        discharge_hold = [t for t in discharge if t.level == 'hold']
        for dd in discharge_deck:
            for dh in discharge_hold:
                prob.phi.append((dd.id, dh.id))

    # Rule 3: Load - hold before deck
        load_hold = [t for t in load if t.level == 'hold']
        load_deck = [t for t in load if t.level == 'deck']
        for lh in load_hold:
            for ld in load_deck:
                prob.phi.append((lh.id, ld.id))

        # ------------------
        # Non-simultaneity Ψ
        # ------------------
        prob.psi = []

        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                if abs(tasks[i].bay - tasks[j].bay) <= 1:
                    prob.psi.append((tasks[i].id, tasks[j].id))

        # ------------------
        # QC travel times
        # ------------------
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
