import psutil
import time


def get_realtime_workload(max_processes=10):
    """
    Collects real-time CPU process data and converts it
    into a simulated scheduling workload.
    """

    processes = []
    now = int(time.time())

    for p in psutil.process_iter(["pid", "cpu_times", "nice", "create_time"]):
        try:
            cpu_time = p.info["cpu_times"].user
            arrival_time = int(p.info["create_time"])
            priority = p.info["nice"]

            remaining_time = max(1, int(cpu_time * 10))  # scaled estimate

            processes.append({
                "pid": p.info["pid"],
                "arrival_time": arrival_time % 100,
                "burst_time": remaining_time,
                "remaining_time": remaining_time,
                "priority": priority,
                "start_time": None,
                "finish_time": None,
            })

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Limit workload size
    processes = sorted(processes, key=lambda p: p["arrival_time"])
    return processes[:max_processes]
