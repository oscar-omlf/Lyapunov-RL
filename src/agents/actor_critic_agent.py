import gymnasium as gym
import torch
import numpy as np

from agents.abstract_agent import AbstractAgent
from models.mlp import MLP
from models.mlpmultivariategaussian import MLPMultivariateGaussian
from models.sampling import sample_two_headed_gaussian_model
from trainers.ac_trainer import ACTrainer
from util.device import fetch_device


class ActorCriticAgent(AbstractAgent):
    def __init__(self, config: dict):
        """
        Initialize the n-step actor-critic agent.
        """
        super().__init__(config)
        
        self.n_steps = config.get("n_steps", 1)
        self.actor_lr = config.get("actor_lr", 1e-3)
        self.critic_lr = config.get("critic_lr", 5e-3)

        # Extract the environment dimensions
        state_dim = config["state_space"].shape[0]
        action_dim = config["action_space"].shape[0] # Assumes continuous action space (like Pendulum-v1)!
        
        # Initialize the actor and critic models
        self._actor_model = MLPMultivariateGaussian(input_size=state_dim, output_size=action_dim).to(device=self.device) # Remember to set to evaluation mode afterwards!!
        # output_size=1 because value function returns a scalar value.
        self._critic_model = MLP(input_size=state_dim, output_size=1).to(device=self.device)

        self._trainer = ACTrainer(
            buffer=self._replay_buffer,
            actor=self._actor_model,
            critic=self._critic_model,
            gamma=self.gamma,
            n_steps=self.n_steps,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
        )

    def add_transition(self, transition: tuple) -> None:
        """
        Add a transition to the replay buffer.

        :param transition: The transition to add to the replay buffer.
        """
        state, action, reward, next_state, done = transition
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        action_t = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        reward_t = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
        next_state_t = torch.as_tensor(next_state, device=self.device, dtype=torch.float32)
        done_t = torch.as_tensor(done, device=self.device, dtype=torch.bool)
        self._replay_buffer.push((state_t, action_t, reward_t, next_state_t, done_t))

    def update(self, flush: bool = False) -> None:
        """
        Perform a gradient descent step on both actor (policy) and critic (value function).
        """
        return self._trainer.train(flush=flush)

    def policy(self, state) -> np.array:
        """
        Get the action to take based on the current state.

        :param state: The current state of the environment.
        :return: The action to take.
        """
        state = torch.from_numpy(state).to(device=self.device, dtype=torch.float32)  # just to make it a Tensor obj

        action, _ = sample_two_headed_gaussian_model(self._actor_model, state)

        return action.cpu().numpy()  # Put the tensor back on the CPU (if applicable) and convert to numpy array.

    def save(self, file_path='./') -> None:
        """
        Save the actor and critic models.

        :param file_path: The directory path to save the models.
        """
        torch.save(self._actor_model.state_dict(), file_path + "actor_model.pth")
        torch.save(self._critic_model.state_dict(), file_path + "critic_model.pth")

    def load(self, file_path='./') -> None:
        """
        Load the actor and critic models.

        :param file_path: The directory path to load the models from.
        """
        self._actor_model.load_state_dict(torch.load(file_path + "actor_model.pth"))
        self._critic_model.load_state_dict(torch.load(file_path + 'critic_model.pth'))
