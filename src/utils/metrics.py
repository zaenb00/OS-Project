def calculate_waiting_time(process):
    """
    Waiting time = time when process first gets CPU - arrival time
    """
    if process["start_time"] is None:
        return 0
    return process["start_time"] - process["arrival_time"]


def calculate_turnaround_time(process):
    """
    Turnaround time = finish time - arrival time
    """
    if process["finish_time"] is None:
        return 0
    return process["finish_time"] - process["arrival_time"]


def average_waiting_time(processes):
    total = sum(calculate_waiting_time(p) for p in processes)
    return total / len(processes)


def average_turnaround_time(processes):
    total = sum(calculate_turnaround_time(p) for p in processes)
    return total / len(processes)


def cpu_utilization(total_time, idle_time):
    """
    CPU Utilization percentage
    """
    if total_time == 0:
        return 0
    busy_time = total_time - idle_time
    return (busy_time / total_time) * 100


if __name__ == "__main__":
    dummy = {
        "arrival_time": 2,
        "start_time": 5,
        "finish_time": 10
    }

    print("Waiting:", calculate_waiting_time(dummy))       # 3
    print("Turnaround:", calculate_turnaround_time(dummy)) # 8
