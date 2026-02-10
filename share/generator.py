"""
Generator for Quay Crane Scheduling Problem (QCSP) instances.

This module defines a function to create random QCSP instances.  Each instance
consists of a set of tasks, each of which has a processing time, a bay
location on the vessel, and optionally precedence constraints within the same
bay.  The number of cranes, the moving time between bays and a safety
distance are also included.

The generator does not rely on any external solver libraries and is designed
to support heuristic benchmarking.  It uses Python's built‑in random module
for reproducible instance generation via a seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Task:
    """Represents a single container handling task on a quay crane.

    Attributes
    ----------
    id : int
        Unique identifier for the task.
    bay : int
        Bay index on the vessel where the task is located.
    processing_time : int
        Time required to complete the task.
    predecessors : List[int]
        List of task IDs that must be processed before this task.
    """

    id: int
    bay: int
    processing_time: int
    predecessors: List[int] = field(default_factory=list)


@dataclass
class Instance:
    """Represents a QCSP instance.

    Attributes
    ----------
    tasks : List[Task]
        List of tasks in the instance.
    n_cranes : int
        Number of available quay cranes.
    move_time : int
        Time required for a crane to move between adjacent bays.
    safety_distance : int
        Number of bays that must separate two cranes to avoid interference.
    max_bay : int
        The highest bay index in the instance (used to determine the vessel
        size).
    """

    tasks: List[Task]
    n_cranes: int
    move_time: int
    safety_distance: int
    max_bay: int


def generate_instance(
    n_tasks: int,
    n_cranes: int,
    max_bay: int,
    seed: int | None = None,
    min_processing: int = 10,
    max_processing: int = 50,
    allow_precedence: bool = True,
) -> Instance:
    """Generate a random QCSP instance.

    Parameters
    ----------
    n_tasks : int
        Total number of tasks to generate.
    n_cranes : int
        Number of cranes available for scheduling.
    max_bay : int
        The highest bay index (bays are numbered from 1 to max_bay).
    seed : int or None
        Seed for the random number generator to ensure reproducibility.
    min_processing : int
        Minimum processing time for each task.
    max_processing : int
        Maximum processing time for each task.
    allow_precedence : bool
        If True, precedence constraints are created within each bay by chaining
        tasks in a random order.

    Returns
    -------
    Instance
        A randomly generated QCSP instance.
    """

    rng = random.Random(seed)
    tasks: List[Task] = []
    # Create tasks with random bays and processing times
    for task_id in range(n_tasks):
        bay = rng.randint(1, max_bay)
        processing_time = rng.randint(min_processing, max_processing)
        tasks.append(Task(id=task_id, bay=bay, processing_time=processing_time))

    if allow_precedence:
        # Group tasks by bay and create precedence chains within each bay
        tasks_by_bay: Dict[int, List[Task]] = {}
        for task in tasks:
            tasks_by_bay.setdefault(task.bay, []).append(task)
        for bay, tlist in tasks_by_bay.items():
            rng.shuffle(tlist)
            for idx, task in enumerate(tlist):
                if idx > 0:
                    # Each task has the previous task in this bay as its predecessor
                    task.predecessors.append(tlist[idx - 1].id)

    # Assign constant move time and safety distance (can be adjusted)
    move_time = 5  # time units to move one bay
    safety_distance = 1  # bays of separation required between cranes

    return Instance(
        tasks=tasks,
        n_cranes=n_cranes,
        move_time=move_time,
        safety_distance=safety_distance,
        max_bay=max_bay,
    )


if __name__ == "__main__":  # Simple demo
    inst = generate_instance(n_tasks=10, n_cranes=3, max_bay=5, seed=42)
    for t in inst.tasks:
        print(t)