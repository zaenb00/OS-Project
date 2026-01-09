from src.utils.workload import generate_workload
from src.env.cpu_env import CPUSchedulingEnv
from src.agents.q_learning import QLearningAgent
import numpy as np


def train(episodes=200):
    total_rewards = []

    for ep in range(episodes):
        processes = generate_workload(num_processes=5, seed=ep)
        env = CPUSchedulingEnv(processes)
        agent = QLearningAgent(action_space_size=env.max_queue_size)

        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    return total_rewards


if __name__ == "__main__":
    rewards = train()
    print("Training completed.")