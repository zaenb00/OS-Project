from src.env.cpu_env import CPUSchedulingEnv
from src.utils.workload import generate_workload


def test_env_state_and_progress():
    processes = generate_workload(num_processes=3, seed=1)
    env = CPUSchedulingEnv(processes, max_queue_size=4)
    state = env.reset()
    # state length = max_queue_size + 1 (queue length)
    assert len(state) == 5

    done = False
    steps = 0
    # Run until done (safety cap)
    while not done and steps < 1000:
        action = 0
        state, reward, done = env.step(action)
        steps += 1

    assert done
    assert env.time >= 0