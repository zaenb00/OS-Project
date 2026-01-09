from src.utils.workload import generate_workload
from src.env.cpu_env import CPUSchedulingEnv
from src.agents.dqn import DQNAgent
import torch
import os


def train(episodes=300, save_path="dqn_scheduler.pth"):
    agent = None  # keep ONE agent

    for ep in range(episodes):
        processes = generate_workload(num_processes=6, seed=ep)
        env = CPUSchedulingEnv(processes)

        if agent is None:
            agent = DQNAgent(
                state_size=len(env.reset()),
                action_size=env.max_queue_size,
                target_update_steps=50,  # faster sync for small experiments
            )

        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

        if (ep + 1) % 50 == 0:
            print(
                f"Episode {ep+1}, Reward: {total_reward:.2f}, "
                f"Epsilon: {agent.epsilon:.2f}"
            )

    # ✅ SAVE TRAINED MODEL
    torch.save(agent.model.state_dict(), save_path)
    print(f"\nDQN model saved to {save_path}")


if __name__ == "__main__":
    train()
    print("Training completed.")