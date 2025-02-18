from abc import ABC, abstractmethod
import torch
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, transition):
        """
        Store a transition tuple (state, action, reward, next_state, done)
        and convert each element to a tensor.
        """
        state, action, reward, next_state, done = transition
        
        state = torch.tensor(state, dtype=torch.float)
        # For continuous actions:
        action = torch.tensor(action, dtype=torch.float)
        # For discrete actions:
        # action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.bool)
        
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """
        Return a random batch of transitions.
        """
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)

class AbstractAgent(ABC):
    def __init__(self, config):
        """
        Initialize the agent with configuration parameters.
        Config should be a dictionary or configuration object that includes:
            - 'state_space'
            - 'action_space'
            - 'gamma' (discount factor)
            - 'buffer_size' for the replay buffer, and any other hyperparameters.
        
        If 'buffer_size' is None, no replay buffer is created.
        """
        self.config = config
        buffer_size = config.get('buffer_size', 500)
        self.replay_buffer = ReplayBuffer(buffer_size)
    
    @abstractmethod
    def policy(self, state):
        """
        Given a state, compute and return an action according to the agent's policy.
        This method replaces the 'act' method for clarity.
        """
        pass

    @abstractmethod
    def update(self):
        """
        Update the agent's parameters based on a batch of transitions sampled from the replay buffer.
        """
        pass

    @abstractmethod
    def save(self, filepath):
        """
        Save the model's parameters or the entire agent state to the specified filepath.
        """
        pass

    @abstractmethod
    def load(self, filepath):
        """
        Load the model's parameters or the entire agent state from the specified filepath.
        """
        pass

    def add_transition(self, transition):
        """
        Store a new transition in the replay buffer.
        Each transition should be a tuple: (state, action, reward, next_state, done).
        """
        if self.replay_buffer is not None:
            self.replay_buffer.push(transition)
