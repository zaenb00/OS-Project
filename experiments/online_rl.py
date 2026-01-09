from src.utils.realtime_cpu import get_realtime_workload
from src.env.cpu_env import CPUSchedulingEnv
from src.agents.dqn import DQNAgent
import time


def online_training(steps=100):
    agent = None

    for step in range(steps):
        processes = get_realtime_workload()

        if not processes:
            continue

        env = CPUSchedulingEnv(processes)

        if agent is None:
            agent = DQNAgent(
                state_size=len(env.reset()),
                action_size=env.max_queue_size,
            )

        state = env.reset()
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state

        if step % 10 == 0:
            print(
                f"Step {step} | "
                f"Epsilon: {agent.epsilon:.2f}"
            )

        time.sleep(1)  # real-time pacing


if __name__ == "__main__":
    online_training()
