#
# class Agent:
#     def __init__(
#         self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
#     ):
#         self.action_space = action_space
#         self.observation_space = observation_space
#
#     def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
#         return self.action_space.sample()
#
#     def learn(
#         self,
#         observation: gym.spaces.Box,
#         reward: float,
#         terminated: bool,
#         truncated: bool,
#     ) -> None:
#
#        pass

import gymnasium as gym
import numpy as np


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
        self.last_action = None

    # def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
    #     if np.random.uniform() < self.exploration_rate:
    #         # Explore - choose a random action
    #         self.last_action = int(self.action_space.sample())
    #     else:
    #         # Exploit - choose the action with highest Q-value
    #         self.last_action = np.argmax(self.q_table[observation])
    #     #print(f"Action: {self.last_action}, Type: {type(gym.spaces.Discrete(self.last_action))}")
    #     return gym.spaces.Discrete(self.last_action % self.action_space.n)
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
        truncated: bool,
        # next_observation: gym.spaces.Box
    ) -> None:
        """
        Learn based on the rewards
        """
        if self.last_action is not None:
            next_observation = observation
            # Update the Q-value for the previous state-action pair
            old_q_value = self.q_table[observation][self.last_action]
            next_max_q_value = np.max(self.q_table[next_observation])
            new_q_value = (
                1 - self.learning_rate
            ) * old_q_value + self.learning_rate * (
                reward + self.discount_factor * next_max_q_value
            )
            self.q_table[observation][self.last_action] = new_q_value

            # Update exploration rate
            self.exploration_rate = max(
                self.min_exploration_rate,
                self.exploration_rate * self.exploration_decay,
            )

    def new_episode(self) -> None:
        # Reset exploration rate and choose initial action
        self.exploration_rate = 1.0
        self.last_action = None


# Define the replay buffer
class ReplayBuffer:
    """
    The replay buffer will store all the experiences.
    """

    def __init__(self, capacity):
        """
        Initialize.
        """
        self.buffer = []
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        """
        Add on new action.
        """
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample it.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )
