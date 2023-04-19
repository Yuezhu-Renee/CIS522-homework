import gymnasium as gym
import numpy as np
import random
from collections import deque


class Agent:
    """
    The agent with act and learn.
    """

    def __init__(
        self,
        action_space: gym.spaces.Discrete,
        observation_space: gym.spaces.Box,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        exploration_rate: float = 1.0,
        exploration_decay: float = 0.99,
        min_exploration_rate: float = 0.01,
        replay_buffer_size: int = 10000,
        batch_size: int = 32,
    ):
        """
        Initialize the parameters.
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        self.q_table = np.zeros((self.observation_space.shape[0], self.action_space.n))
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.batch_size = batch_size
        self.last_observation = None
        self.last_action = None

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        Act based on interaction.
        """
        if observation in self.q_table:
            max_action = np.argmax(self.q_table[observation])
            return max_action
        else:
            return self.action_space.sample()

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool
        # next_observation: gym.spaces.Box
    ) -> None:
        """
        Learn based on the rewards
        """
        next_observation = gym.spaces.Box
        if self.last_action is not None:
            observation_idx = np.ravel_multi_index(
                observation.astype(int), self.observation_space.shape
            )
            last_action_idx = int(self.last_action)
            next_observation_idx = np.ravel_multi_index(
                next_observation.astype(int), self.observation_space.shape
            )

            # Update the Q-value for the previous state-action pair
            old_q_value = self.q_table[observation_idx, last_action_idx]
            next_max_q_value = np.max(self.q_table[next_observation_idx])
            new_q_value = (
                1 - self.learning_rate
            ) * old_q_value + self.learning_rate * (
                reward + self.discount_factor * next_max_q_value
            )
            self.q_table[observation_idx, last_action_idx] = new_q_value

            # Update exploration rate
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay,
            )

    def new_episode(self) -> None:
        # Reset exploration rate and choose initial action
        self.exploration_rate = 1.0
        self.last_observation = None
        self.last_action = None


class ReplayBuffer:
    """
    A buffer for storing and retrieving experiences (state, action, reward, next_state, done).
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to the buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for state, action, reward, next_state, done in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Return the number of experiences in the buffer.
        """
        return len(self.buffer)
