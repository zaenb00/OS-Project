import os
import copy
from src.env.cpu_env import CPUSchedulingEnv
from src.agents.dqn import DQNAgent


def rl_dqn_scheduler(
    processes,
    num_cores=1,
    episodes=200,
    model_path="models/pretrained_rl_scheduler.pt",
):
    """
    Offline RL Scheduler (Multi-core)
    --------------------------------
    ✔ Loads pretrained model if compatible
    ✔ Retrains if incompatible or missing
    ✔ Uses action masking
    ✔ Saves updated model with metadata
    """

    # -------------------------------------------------
    # Environment
    # -------------------------------------------------
    env = CPUSchedulingEnv(copy.deepcopy(processes), num_cores=num_cores)

    state = env.reset()
    state_size = len(state)
    action_size = env.max_queue_size * num_cores

    agent = DQNAgent(state_size, action_size)

    # -------------------------------------------------
    # Load pretrained model (SAFE)
    # -------------------------------------------------
    if os.path.exists(model_path):
        try:
            agent.load(
                model_path,
                state_size=state_size,
                action_size=action_size,
                num_cores=num_cores,
            )
            print("✅ Pretrained RL model loaded")
        except Exception as e:
            print(f"⚠️ Pretrained model incompatible, retraining... ({e})")

    # -------------------------------------------------
    # Offline Training Loop
    # -------------------------------------------------
    for _ in range(episodes):
        state = env.reset()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()

            if not valid_actions:
                next_state, reward, done = env.step(None)
                action = None
            else:
                action = agent.act(state, valid_actions)
                next_state, reward, done = env.step(action)

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()

    # -------------------------------------------------
    # Save updated model
    # -------------------------------------------------
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    agent.save(
        model_path,
        num_cores=num_cores,
    )

    # -------------------------------------------------
    # Results
    # -------------------------------------------------
    completed = env.completed
    total_time = env.time
    idle_time = sum(
        1 for _ in range(env.time * num_cores)
    ) - sum(env.core_busy_time)

    return completed, total_time, idle_time
