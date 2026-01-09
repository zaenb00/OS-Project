import numpy as np

class CPUSchedulingEnv:
    def __init__(self, processes, num_cores=1):
        self.original_processes = processes
        self.num_cores = num_cores
        self.max_queue_size = len(processes)
        self.reset()

    def reset(self):
        self.time = 0

        self.processes = []
        for i, p in enumerate(self.original_processes):
            proc = p.copy()
            proc["pid"] = i
            proc["remaining_time"] = proc["burst_time"]
            proc["start_time"] = None
            proc["finish_time"] = None
            proc["core"] = None
            self.processes.append(proc)

        self.ready_queue = []
        self.cores = [None] * self.num_cores
        self.completed = []
        self.core_busy_time = [0] * self.num_cores

        return self._get_state()

    def _get_state(self):
        core_states = [c["remaining_time"] if c else 0 for c in self.cores]
        queue_len = len(self.ready_queue)
        avg_wait = np.mean(
            [self.time - p["arrival_time"] for p in self.ready_queue]
        ) if self.ready_queue else 0

        return np.array(core_states + [queue_len, avg_wait], dtype=np.float32)

    def get_valid_actions(self):
        valid = []
        for p_idx in range(len(self.ready_queue)):
            for c_idx in range(self.num_cores):
                if self.cores[c_idx] is None:
                    valid.append(p_idx * self.num_cores + c_idx)
        return valid

    def step(self, action):
        reward = 0
        done = False

        for p in self.processes:
            if p["arrival_time"] == self.time:
                self.ready_queue.append(p)

        if action is not None:
            p_idx = action // self.num_cores
            c_idx = action % self.num_cores

            if (
                p_idx < len(self.ready_queue)
                and c_idx < self.num_cores
                and self.cores[c_idx] is None
            ):
                proc = self.ready_queue.pop(p_idx)
                proc["core"] = c_idx
                if proc["start_time"] is None:
                    proc["start_time"] = self.time
                self.cores[c_idx] = proc

        self.time += 1

        idle = 0
        for i in range(self.num_cores):
            core = self.cores[i]
            if core:
                core["remaining_time"] -= 1
                self.core_busy_time[i] += 1
                if core["remaining_time"] <= 0:
                    core["finish_time"] = self.time
                    self.completed.append(core)
                    self.cores[i] = None
                    reward += 10
            else:
                idle += 1

        reward -= len(self.ready_queue)
        reward -= idle

        if (
            len(self.completed) == len(self.processes)
            and not self.ready_queue
            and all(c is None for c in self.cores)
        ):
            done = True

        return self._get_state(), reward, done
import numpy as np

class CPUSchedulingEnv:
    def __init__(self, processes, num_cores=1):
        self.original_processes = processes
        self.num_cores = num_cores
        self.max_queue_size = len(processes)
        self.reset()

    def reset(self):
        self.time = 0

        self.processes = []
        for i, p in enumerate(self.original_processes):
            proc = p.copy()
            proc["pid"] = i
            proc["remaining_time"] = proc["burst_time"]
            proc["start_time"] = None
            proc["finish_time"] = None
            proc["core"] = None
            self.processes.append(proc)

        self.ready_queue = []
        self.cores = [None] * self.num_cores
        self.completed = []
        self.core_busy_time = [0] * self.num_cores

        return self._get_state()

    def _get_state(self):
        core_states = [c["remaining_time"] if c else 0 for c in self.cores]
        queue_len = len(self.ready_queue)
        avg_wait = np.mean(
            [self.time - p["arrival_time"] for p in self.ready_queue]
        ) if self.ready_queue else 0

        return np.array(core_states + [queue_len, avg_wait], dtype=np.float32)

    def get_valid_actions(self):
        valid = []
        for p_idx in range(len(self.ready_queue)):
            for c_idx in range(self.num_cores):
                if self.cores[c_idx] is None:
                    valid.append(p_idx * self.num_cores + c_idx)
        return valid

    def step(self, action):
        reward = 0
        done = False

        for p in self.processes:
            if p["arrival_time"] == self.time:
                self.ready_queue.append(p)

        if action is not None:
            p_idx = action // self.num_cores
            c_idx = action % self.num_cores

            if (
                p_idx < len(self.ready_queue)
                and c_idx < self.num_cores
                and self.cores[c_idx] is None
            ):
                proc = self.ready_queue.pop(p_idx)
                proc["core"] = c_idx
                if proc["start_time"] is None:
                    proc["start_time"] = self.time
                self.cores[c_idx] = proc

        self.time += 1

        idle = 0
        for i in range(self.num_cores):
            core = self.cores[i]
            if core:
                core["remaining_time"] -= 1
                self.core_busy_time[i] += 1
                if core["remaining_time"] <= 0:
                    core["finish_time"] = self.time
                    self.completed.append(core)
                    self.cores[i] = None
                    reward += 10
            else:
                idle += 1

        reward -= len(self.ready_queue)
        reward -= idle

        if (
            len(self.completed) == len(self.processes)
            and not self.ready_queue
            and all(c is None for c in self.cores)
        ):
            done = True

        return self._get_state(), reward, done
import numpy as np

class CPUSchedulingEnv:
    def __init__(self, processes, num_cores=1):
        self.original_processes = processes
        self.num_cores = num_cores
        self.max_queue_size = len(processes)
        self.reset()

    def reset(self):
        self.time = 0

        self.processes = []
        for i, p in enumerate(self.original_processes):
            proc = p.copy()
            proc["pid"] = i
            proc["remaining_time"] = proc["burst_time"]
            proc["start_time"] = None
            proc["finish_time"] = None
            proc["core"] = None
            self.processes.append(proc)

        self.ready_queue = []
        self.cores = [None] * self.num_cores
        self.completed = []
        self.core_busy_time = [0] * self.num_cores

        return self._get_state()

    def _get_state(self):
        core_states = [c["remaining_time"] if c else 0 for c in self.cores]
        queue_len = len(self.ready_queue)
        avg_wait = np.mean(
            [self.time - p["arrival_time"] for p in self.ready_queue]
        ) if self.ready_queue else 0

        return np.array(core_states + [queue_len, avg_wait], dtype=np.float32)

    def get_valid_actions(self):
        valid = []
        for p_idx in range(len(self.ready_queue)):
            for c_idx in range(self.num_cores):
                if self.cores[c_idx] is None:
                    valid.append(p_idx * self.num_cores + c_idx)
        return valid

    def step(self, action):
        reward = 0
        done = False

        for p in self.processes:
            if p["arrival_time"] == self.time:
                self.ready_queue.append(p)

        if action is not None:
            p_idx = action // self.num_cores
            c_idx = action % self.num_cores

            if (
                p_idx < len(self.ready_queue)
                and c_idx < self.num_cores
                and self.cores[c_idx] is None
            ):
                proc = self.ready_queue.pop(p_idx)
                proc["core"] = c_idx
                if proc["start_time"] is None:
                    proc["start_time"] = self.time
                self.cores[c_idx] = proc

        self.time += 1

        idle = 0
        for i in range(self.num_cores):
            core = self.cores[i]
            if core:
                core["remaining_time"] -= 1
                self.core_busy_time[i] += 1
                if core["remaining_time"] <= 0:
                    core["finish_time"] = self.time
                    self.completed.append(core)
                    self.cores[i] = None
                    reward += 10
            else:
                idle += 1

        reward -= len(self.ready_queue)
        reward -= idle

        if (
            len(self.completed) == len(self.processes)
            and not self.ready_queue
            and all(c is None for c in self.cores)
        ):
            done = True

        return self._get_state(), reward, done
import numpy as np

class CPUSchedulingEnv:
    def __init__(self, processes, num_cores=1):
        self.original_processes = processes
        self.num_cores = num_cores
        self.max_queue_size = len(processes)
        self.reset()

    def reset(self):
        self.time = 0

        self.processes = []
        for i, p in enumerate(self.original_processes):
            proc = p.copy()
            proc["pid"] = i
            proc["remaining_time"] = proc["burst_time"]
            proc["start_time"] = None
            proc["finish_time"] = None
            proc["core"] = None
            self.processes.append(proc)

        self.ready_queue = []
        self.cores = [None] * self.num_cores
        self.completed = []
        self.core_busy_time = [0] * self.num_cores

        return self._get_state()

    def _get_state(self):
        core_states = [c["remaining_time"] if c else 0 for c in self.cores]
        queue_len = len(self.ready_queue)
        avg_wait = np.mean(
            [self.time - p["arrival_time"] for p in self.ready_queue]
        ) if self.ready_queue else 0

        return np.array(core_states + [queue_len, avg_wait], dtype=np.float32)

    def get_valid_actions(self):
        valid = []
        for p_idx in range(len(self.ready_queue)):
            for c_idx in range(self.num_cores):
                if self.cores[c_idx] is None:
                    valid.append(p_idx * self.num_cores + c_idx)
        return valid

    def step(self, action):
        reward = 0
        done = False

        for p in self.processes:
            if p["arrival_time"] == self.time:
                self.ready_queue.append(p)

        if action is not None:
            p_idx = action // self.num_cores
            c_idx = action % self.num_cores

            if (
                p_idx < len(self.ready_queue)
                and c_idx < self.num_cores
                and self.cores[c_idx] is None
            ):
                proc = self.ready_queue.pop(p_idx)
                proc["core"] = c_idx
                if proc["start_time"] is None:
                    proc["start_time"] = self.time
                self.cores[c_idx] = proc

        self.time += 1

        idle = 0
        for i in range(self.num_cores):
            core = self.cores[i]
            if core:
                core["remaining_time"] -= 1
                self.core_busy_time[i] += 1
                if core["remaining_time"] <= 0:
                    core["finish_time"] = self.time
                    self.completed.append(core)
                    self.cores[i] = None
                    reward += 10
            else:
                idle += 1

        reward -= len(self.ready_queue)
        reward -= idle

        if (
            len(self.completed) == len(self.processes)
            and not self.ready_queue
            and all(c is None for c in self.cores)
        ):
            done = True

        return self._get_state(), reward, done
