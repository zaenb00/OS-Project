from src.utils.workload import generate_workload
from src.utils.metrics import (
    average_waiting_time,
    average_turnaround_time,
    cpu_utilization,
)
from src.schedulers.fcfs import fcfs_scheduler
from src.schedulers.sjf import sjf_scheduler
from src.schedulers.round_robin import round_robin_scheduler
import copy
import os
from src.schedulers.rl_schedular import rl_dqn_scheduler


def evaluate_all(num_processes=10, seed=42):
    workload = generate_workload(
        num_processes=num_processes,
        seed=seed
    )

    results = {}

    # FCFS
    fcfs_procs = copy.deepcopy(workload)
    completed, total_time, idle_time = fcfs_scheduler(fcfs_procs)
    results["FCFS"] = {
        "avg_waiting": average_waiting_time(completed),
        "avg_turnaround": average_turnaround_time(completed),
        "cpu_util": cpu_utilization(total_time, idle_time),
    }

    # SJF
    sjf_procs = copy.deepcopy(workload)
    completed, total_time, idle_time = sjf_scheduler(sjf_procs)
    results["SJF"] = {
        "avg_waiting": average_waiting_time(completed),
        "avg_turnaround": average_turnaround_time(completed),
        "cpu_util": cpu_utilization(total_time, idle_time),
    }

    # Round Robin
    rr_procs = copy.deepcopy(workload)
    completed, total_time, idle_time = round_robin_scheduler(rr_procs, quantum=2)
    results["Round Robin"] = {
        "avg_waiting": average_waiting_time(completed),
        "avg_turnaround": average_turnaround_time(completed),
        "cpu_util": cpu_utilization(total_time, idle_time),
    }

    # DQN (inference using saved model)
    model_path = "dqn_scheduler.pth"
    if os.path.exists(model_path):
        try:
            dqn_procs = copy.deepcopy(workload)
            completed, total_time, idle_time = rl_dqn_scheduler(dqn_procs, model_path=model_path)
            results["DQN"] = {
                "avg_waiting": average_waiting_time(completed),
                "avg_turnaround": average_turnaround_time(completed),
                "cpu_util": cpu_utilization(total_time, idle_time),
            }
        except Exception as e:
            print("Error running DQN evaluation:", e)
    else:
        print(f"Skipping DQN evaluation - model not found at {model_path}")

    return results


if __name__ == "__main__":
    results = evaluate_all()

    print("\nScheduler Comparison Results\n")
    for algo, metrics in results.items():
        print(f"{algo}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f}")
        print()