def fcfs_scheduler(processes, num_cores=1):
    time = 0
    ready_queue = []
    completed = []

    # Per-core state
    cores = [None] * num_cores
    idle_time = 0

    while len(completed) < len(processes):
        # Add arrivals
        for p in processes:
            if p["arrival_time"] == time:
                ready_queue.append(p)

        # Assign processes to free cores
        for i in range(num_cores):
            if cores[i] is None and ready_queue:
                proc = ready_queue.pop(0)
                if proc["start_time"] is None:
                    proc["start_time"] = time
                proc["core"] = i
                cores[i] = proc


        # Execute
        busy = False
        for i in range(num_cores):
            if cores[i]:
                busy = True
                cores[i]["remaining_time"] -= 1
                if cores[i]["remaining_time"] == 0:
                    cores[i]["finish_time"] = time + 1
                    completed.append(cores[i])
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
        raise SystemExit("Please run this module as a package: 'python -m src.schedulers.fcfs' from the project root (not by running the .py file directly).")

    processes = generate_workload(
        num_processes=5,
        seed=42,
        max_arrival_time=10,
        max_burst_time=10,
    )

    completed, total_time, idle_time = fcfs_scheduler(processes, num_cores=2)

    print("Completed Processes:")
    for p in completed:
        print(
            f"Process ID: {p['id']}, Arrival Time: {p['arrival_time']}, Burst Time: {p['burst_time']}, Start Time: {p['start_time']}, Finish Time: {p['finish_time']}"
        )

    print(f"\nTotal Time: {total_time}")
    print(f"Idle Time: {idle_time}")
    print(f"Average Waiting Time: {average_waiting_time(completed)}")
    print(f"Average Turnaround Time: {average_turnaround_time(completed)}")