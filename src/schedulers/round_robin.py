from collections import deque

def round_robin_scheduler(processes, quantum=2, num_cores=1):
    time = 0
    ready_queue = deque()
    completed = []

    cores = [None] * num_cores
    core_quantum = [0] * num_cores
    idle_time = 0

    while len(completed) < len(processes):
        # Add arrivals
        for p in processes:
            if p["arrival_time"] == time:
                ready_queue.append(p)

        # Assign to free cores
        for i in range(num_cores):
            if cores[i] is None and ready_queue:
                proc = ready_queue.popleft()
                if proc["start_time"] is None:
                    proc["start_time"] = time
                proc["core"] = i
                cores[i] = proc
                core_quantum[i] = quantum

        busy = False
        for i in range(num_cores):
            proc = cores[i]
            if proc:
                busy = True
                proc["remaining_time"] -= 1
                core_quantum[i] -= 1

                if proc["remaining_time"] == 0:
                    proc["finish_time"] = time + 1
                    completed.append(proc)
                    cores[i] = None

                elif core_quantum[i] == 0:
                    ready_queue.append(proc)
                    cores[i] = None

        if not busy:
            idle_time += 1

        time += 1

    return completed, time, idle_time

if __name__ == "__main__":
    try:
        from ..utils.workload import generate_workload
        from ..utils.metrics import average_waiting_time, average_turnaround_time
    except Exception:
        raise SystemExit("Please run this module as a package: 'python -m src.schedulers.round_robin' from the project root (not by running the .py file directly).")

    processes = generate_workload(
        num_processes=5,
        seed=42,
        max_arrival_time=10,
        max_burst_time=10,
    )

    completed, total_time, idle_time = round_robin_scheduler(processes, quantum=3, num_cores=2)

    print("Completed Processes:")
    for p in completed:
        print(
            f"Process ID: {p['id']}, Arrival Time: {p['arrival_time']}, Burst Time: {p['burst_time']}, "
            f"Start Time: {p['start_time']}, Finish Time: {p['finish_time']}, Waiting Time: {p['finish_time'] - p['arrival_time'] - p['burst_time']}, "
            f"Turnaround Time: {p['finish_time'] - p['arrival_time']}"
        )

    print(f"\nTotal Time: {total_time}")
    print(f"Idle Time: {idle_time}")
    print(f"Average Waiting Time: {average_waiting_time(completed)}")
    print(f"Average Turnaround Time: {average_turnaround_time(completed)}")