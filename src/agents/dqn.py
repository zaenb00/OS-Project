import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        memory_size=5000,
        batch_size=32,
        target_update_steps=100,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_network()

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0
        self.target_update_steps = target_update_steps

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state, valid_actions=None):
        if random.random() < self.epsilon:
            return random.choice(valid_actions) if valid_actions else random.randint(0, self.action_size - 1)

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.model(state).squeeze()

        if valid_actions:
            mask = torch.full_like(q, -1e9)
            mask[valid_actions] = q[valid_actions]
            return int(mask.argmax().item())

        return int(q.argmax().item())

    def remember(self, state, action, reward, next_state, done):
        # Do not store invalid actions
        if action is None or action < 0:
            return
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # 🔒 Filter out invalid actions (safety net)
        valid_batch = [
            (s, a, r, ns, d)
            for (s, a, r, ns, d) in self.memory
            if a is not None and a >= 0 and a < self.action_size
        ]

        if len(valid_batch) < self.batch_size:
            return

        batch = random.sample(valid_batch, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.model(states).gather(1, actions).squeeze()

        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0]

        targets = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.learn_step += 1
        if self.learn_step % self.target_update_steps == 0:
            self.update_target_network()


    def freeze(self):
        self.epsilon = 0.0
        for p in self.model.parameters():
            p.requires_grad = False

    def save(self, path, num_cores):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_size": self.state_size,
            "action_size": self.action_size,
            "num_cores": num_cores,
            "model_state": self.model.state_dict()
        }, path)

    def clear_memory(self):
        self.memory.clear()

    def load(self, path, state_size, action_size, num_cores):
        if not os.path.exists(path):
            return False

        checkpoint = torch.load(path, map_location="cpu")

        if (
            checkpoint["state_size"] != state_size
            or checkpoint["action_size"] != action_size
            or checkpoint["num_cores"] != num_cores
        ):
            return False

        self.model.load_state_dict(checkpoint["model_state"])
        self.update_target_network()

        # 🔥 CRITICAL: clear contaminated memory
        self.clear_memory()

        return True
