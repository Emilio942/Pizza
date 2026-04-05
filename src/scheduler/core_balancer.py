import json
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Task:
    id: str
    execution_time_us: float  # c_i
    memory_footprint_b: int   # m_i
    dependencies: List[str]   # DAG edges

class CoreBalancer:
    """
    Solves the Multi-Capacity Bin-Packing Problem for scheduling 
    the clustered_s30_int4 model on the RP2040's Dual Cortex-M0+.
    """
    def __init__(self, config_path: str = 'config/hardware_math_params.json'):
        self.config = self._load_config(config_path)
        self.core_0_mem_limit = self.config['rp2040_math']['sram_limit_kb'] * 1024 / 2
        self.core_1_mem_limit = self.config['rp2040_math']['sram_limit_kb'] * 1024 / 2
        self.bus_arbitration_delay_us = self.config['rp2040_math']['bus_arbitration_delay_us']

    def _load_config(self, path: str) -> dict:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Hardware parameters {path} not found.")
            return {}

    def schedule_dag(self, tasks: List[Task]) -> Dict[str, List[Task]]:
        """
        Approximation algorithm for makespan (C_max) minimization.
        Returns a mapping of {'core_0': [...], 'core_1': [...]}.
        """
        # Sort tasks topologically (mocked here by assuming input is sorted)
        # Then apply greedy bin-packing considering memory and execution time
        schedule = {'core_0': [], 'core_1': []}
        core_0_mem, core_0_time = 0, 0
        core_1_mem, core_1_time = 0, 0

        for task in tasks:
            # Simple heuristic: balance execution time, respect memory
            if core_0_time <= core_1_time and (core_0_mem + task.memory_footprint_b) <= self.core_0_mem_limit:
                schedule['core_0'].append(task)
                core_0_mem += task.memory_footprint_b
                core_0_time += task.execution_time_us
            elif (core_1_mem + task.memory_footprint_b) <= self.core_1_mem_limit:
                schedule['core_1'].append(task)
                core_1_mem += task.memory_footprint_b
                core_1_time += task.execution_time_us
            else:
                logger.warning(f"Task {task.id} cannot be scheduled due to memory constraints.")
        
        logger.info(f"Core 0 time: {core_0_time}us, Core 1 time: {core_1_time}us")
        return schedule

if __name__ == "__main__":
    # Example usage
    balancer = CoreBalancer()
    dummy_tasks = [
        Task("conv1", 1200, 4096, []),
        Task("pool1", 400, 1024, ["conv1"]),
        Task("conv2", 2000, 8192, ["pool1"]),
        Task("fc1", 800, 2048, ["conv2"])
    ]
    balancer.schedule_dag(dummy_tasks)
