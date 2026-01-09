import random
import numpy as np


def generate_workload(
    num_processes: int,
    max_arrival_time: int = 10,
    min_burst: int = 1,
    max_burst: int = 10,
    priority_levels: int = 5,
    seed: int | None = None,
):
    """
    Generates a list of simulated processes.

    Args:
        num_processes (int): Number of processes to generate
        max_arrival_time (int): Max arrival time (0 .. max_arrival_time)
        min_burst (int): Minimum CPU burst time
        max_burst (int): Maximum CPU burst time
        priority_levels (int): Number of priority levels (lower = higher priority)
        seed (int | None): Random seed for reproducibility

    Returns:
        List[dict]: List of process control blocks (PCBs)
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    processes = []

    for pid in range(num_processes):
        arrival_time = random.randint(0, max_arrival_time)
        burst_time = random.randint(min_burst, max_burst)
        priority = random.randint(1, priority_levels)

        process = {
            "pid": pid,
            "arrival_time": arrival_time,
            "burst_time": burst_time,
            "remaining_time": burst_time,
            "priority": priority,
            "start_time": None,
            "finish_time": None,
        }

        processes.append(process)

    # Sort processes by arrival time (important!)
    processes.sort(key=lambda p: p["arrival_time"])

    return processes


def generate_poisson_workload(
    num_processes: int,
    arrival_rate: float = 1.0,
    min_burst: int = 1,
    max_burst: int = 10,
    priority_levels: int = 5,
    seed: int | None = None,
):
    """
    Generates workload where arrivals follow a Poisson process.
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    processes = []
    current_time = 0

    for pid in range(num_processes):
        inter_arrival = np.random.poisson(lam=arrival_rate)
        current_time += inter_arrival

        burst_time = random.randint(min_burst, max_burst)
        priority = random.randint(1, priority_levels)

        process = {
            "pid": pid,
            "arrival_time": current_time,
            "burst_time": burst_time,
            "remaining_time": burst_time,
            "priority": priority,
            "start_time": None,
            "finish_time": None,
        }

        processes.append(process)

    return processes


if __name__ == "__main__":
    workload = generate_workload(
        num_processes=5,
        seed=42
    )

    for p in workload:
        print(p)