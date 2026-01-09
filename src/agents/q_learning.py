import random
from collections import defaultdict
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        action_space_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
    ):
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state -> action values
        self.q_table = defaultdict(lambda: np.zeros(action_space_size))

    def get_action(self, state):
        state_key = tuple(state)

        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)

        return int(np.argmax(self.q_table[state_key]))

    def update(self, state, action, reward, next_state, done):
        state_key = tuple(state)
        next_state_key = tuple(next_state)

        best_next = np.max(self.q_table[next_state_key])
        target = reward + (0 if done else self.gamma * best_next)

        self.q_table[state_key][action] += self.lr * (
            target - self.q_table[state_key][action]
        )

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
