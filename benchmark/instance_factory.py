import random
from typing import List, Tuple, Dict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qc_problem import QCProblem


class InstanceFactory:
    def __init__(self, config: Dict):
        self.config = config
        self.instances_generated = []
    
    def generate(
        self,
        num_tasks: int,
        num_qcs: int,
        seed: int
    ) -> Tuple[QCProblem, Dict]:
        # Determine num_bays
        if self.config.get("num_bays_mode") == "proportional":
            num_bays = num_tasks
        else:
            num_bays = self.config.get("fixed_bays", 30)
        
        # Generate instance
        problem = QCProblem.generate(
            num_bays=num_bays,
            num_qcs=num_qcs,
            total_tasks=num_tasks,
            processing_time_range=tuple(self.config["processing_time_range"]),
            travel_per_bay=self.config["travel_per_bay"],
            precedence_probability=self.config["precedence_probability"],
            seed=seed
        )
        
        # Metadata
        metadata = {
            "num_tasks": num_tasks,
            "num_qcs": num_qcs,
            "num_bays": num_bays,
            "seed": seed,
            "processing_time_range": self.config["processing_time_range"],
            "travel_per_bay": self.config["travel_per_bay"],
            "precedence_probability": self.config["precedence_probability"],
        }
        
        self.instances_generated.append(metadata)
        return problem, metadata
    
    def generate_suite(
        self,
        task_sizes: List[int],
        qc_counts: List[int],
        num_instances_per_config: int,
        seed_base: int = 42
    ) -> List[Tuple[QCProblem, Dict]]:
        instances = []
        instance_id = 0
        
        for num_tasks in task_sizes:
            for num_qcs in qc_counts:
                for i in range(num_instances_per_config):
                    seed = seed_base + instance_id
                    problem, metadata = self.generate(num_tasks, num_qcs, seed)
                    metadata["instance_id"] = instance_id
                    metadata["config_id"] = f"T{num_tasks}_K{num_qcs}"
                    instances.append((problem, metadata))
                    instance_id += 1
        
        return instances
