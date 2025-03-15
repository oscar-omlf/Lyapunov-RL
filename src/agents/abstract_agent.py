from abc import ABC, abstractmethod
import gymnasium as gym

from util.device import fetch_device

class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        
    def push(self, transition):
        self.buffer.append(transition)

    def clear(self):
        self.buffer.clear()

    def popleft(self):
        return self.buffer.popleft()

    def get_buffer_list(self):
        return list(self.buffer)
        
    def __len__(self):
        return len(self.buffer)

class AbstractAgent(ABC):
    def __init__(self, config: dict):

        self.state_space = config.get("state_space", 3)
        self.action_space = config.get("action_space", 1)
        self.device = fetch_device()
        self._replay_buffer = ReplayBuffer()

        print(f'Device: {self.device}')

    @abstractmethod
    def add_transition(self, transition):
        """
        Abstract method to add a transition to the agent's replay buffer.

        :param transition: transition to add to the replay buffer.
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """
        Abstract method where the update rule is applied.
        """
        pass

    @abstractmethod
    def policy(self, state):
        """
        Abstract method to define the agent's policy.
        For actor critic algorithms the output of the actor would be a probability distribution over actions.
        For discrete actions this is simply a discrete probability distributions, describing a probability
        for each action.
        For continuous actions you can have some kind of continuous distribution you sample actions from.

        :param state: The current state of the environment.
        """
        pass

    @abstractmethod
    def save(self, file_path: str = './') -> None:
        """
        Abstract method to save the agent's model.

        :param file_path: The path to save the model.
        """
        pass

    @abstractmethod
    def load(self, file_path: str = './') -> None:
        """
        Abstract method to load the agent's model.

        :param file_path: The path to load the model from.
        """
        pass
