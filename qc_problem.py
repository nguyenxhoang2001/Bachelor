import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass(frozen=True)
class Task:
    id: int
    bay: int
    duration: float

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
        precedence_probability: float = 0.3,
        seed: int = None
    ):
        if seed is not None:
            random.seed(seed)

        prob = cls()
        prob.qcs = list(range(1, num_qcs + 1))
        tasks: List[Task] = []

        for tid in range(1, total_tasks + 1):
            bay = random.randint(1, num_bays)
            duration = random.uniform(*processing_time_range)

            tasks.append(
                Task(
                    id=tid,
                    bay=bay,
                    duration=duration
                )
            )

        tasks.sort(key=lambda t: t.bay)
        id_map = {old.id: i + 1 for i, old in enumerate(tasks)}

        tasks = [
            Task(
                id=id_map[t.id],
                bay=t.bay,
                duration=t.duration
            )
            for t in tasks
        ]

        prob.tasks = [t.id for t in tasks]
        prob.duration = {t.id: t.duration for t in tasks}
        prob.location = {t.id: t.bay for t in tasks}
        prob.phi = []
        by_bay: Dict[int, List[Task]] = {}

        for t in tasks:
            by_bay.setdefault(t.bay, []).append(t)

        for bay, bay_tasks in by_bay.items():
            for i in range(len(bay_tasks)):
                for j in range(i + 1, len(bay_tasks)):
                    if random.random() < precedence_probability:
                        prob.phi.append((bay_tasks[i].id, bay_tasks[j].id))

        prob.psi = []
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                if abs(tasks[i].bay - tasks[j].bay) == 1:
                    prob.psi.append((tasks[i].id, tasks[j].id))

        prob.earliest_time = {k: 0.0 for k in prob.qcs}
        prob.initial_position = {}
        for k in prob.qcs:
            if num_qcs > 1:
                position = int((k - 1) * (num_bays - 1) / (num_qcs - 1)) + 1
            else:
                position = 1
            prob.initial_position[k] = position
        
        prob.final_position = {k: prob.initial_position[k] for k in prob.qcs}

        prob.starting_travel_time = {}
        prob.travel_time = {}
        prob.final_travel_time = {}

        for k in prob.qcs:
            start_bay = prob.initial_position[k]
            end_bay = prob.final_position[k]

            for t in tasks:
                prob.starting_travel_time[(k, t.id)] = travel_per_bay * abs(start_bay - t.bay)

            for ti in tasks:
                for tj in tasks:
                    prob.travel_time[(k, ti.id, tj.id)] = travel_per_bay * abs(ti.bay - tj.bay)

            for t in tasks:
                prob.final_travel_time[(k, t.id)] = travel_per_bay * abs(t.bay - end_bay)
        
        total_processing_time = sum(prob.duration.values())
        max_travel_time = travel_per_bay * (num_bays - 1) * len(tasks)
        prob.M = total_processing_time + max_travel_time
        
        prob.alpha1 = 100
        prob.alpha2 = 1

        return prob
