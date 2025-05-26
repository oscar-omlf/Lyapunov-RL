import os
import torch
import numpy as np

from agents.abstract_agent import AbstractAgent
from models.ac_critic import ACCritic
from models.ac_actor import ACActor
from util.sampling import sample_two_headed_gaussian_model
from trainers.ac_trainer import ACTrainer


class ActorCriticAgent(AbstractAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.gamma = config.get("gamma")
        self.n_steps = config.get("n_steps")
        self.policy_freq = config.get("policy_freq")
        self.actor_lr = config.get("actor_lr")
        self.critic_lr = config.get("critic_lr")

        # Extract environment dimensions
        state_dim = self.state_space.shape[0]
        action_dim = self.action_space.shape[0]

        # Get architecture hyperparameters from config
        actor_hidden_sizes = config.get("actor_hidden_sizes", (64, 64))
        critic_hidden_sizes = config.get("critic_hidden_sizes", (64, 64))
        
        # Initialize the actor and critic models with the tunable architectures
        self.actor_model = ACActor(input_size=state_dim,
                                    hidden_sizes=actor_hidden_sizes,
                                    action_dim=action_dim).to(device=self.device)
        
        self.critic_model = ACCritic(input_size=state_dim,
                                      hidden_sizes=critic_hidden_sizes).to(device=self.device)

        self.trainer = ACTrainer(
            buffer=self._replay_buffer,
            actor=self.actor_model,
            critic=self.critic_model,
            gamma=self.gamma,
            n_steps=self.n_steps,
            policy_freq=self.policy_freq,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            device=self.device
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

    def update(self) -> None:
        """
        Perform a gradient descent step on both actor (policy) and critic (value function).
        """
        return self.trainer.train()

    def policy(self, state) -> np.array:
        """
        Get the action to take based on the current state.

        :param state: The current state of the environment.
        :return: The action to take.
        """
        state = torch.from_numpy(state).to(device=self.device, dtype=torch.float32)

        action, _ = sample_two_headed_gaussian_model(self.actor_model, state)

        return action.cpu().numpy()

    def save(self, file_path='./saved_models/') -> None:
        """
        Save the actor and critic models.

        :param file_path: The directory path to save the models.
        """
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.actor_model.state_dict(), file_path + "ac_actor_model.pth")
        torch.save(self.critic_model.state_dict(), file_path + "ac_critic_model.pth")

    def load(self, file_path='./saved_models/') -> None:
        """
        Load the actor and critic models.

        :param file_path: The directory path to load the models from.
        """
        self.actor_model.load_state_dict(torch.load(file_path + "ac_actor_model.pth", weights_only=True))
        self.critic_model.load_state_dict(torch.load(file_path + "ac_critic_model.pth"))
