import os
import copy
import torch
import numpy as np

from agents.abstract_agent import AbstractAgent
from models.td3_actor import TD3Actor
from models.td3_critic import TD3Critic
from trainers.td3_trainer import TD3Trainer


class TD3Agent(AbstractAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.max_action = float(self.action_space.high[0])
        
        self.gamma = config.get("gamma")
        self.tau = config.get("tau")
        self.policy_freq = config.get("policy_freq")
        self.batch_size = config.get("batch_size")

        # Noise used in target updates (for training)
        self.policy_noise = config.get("policy_noise") * self.max_action
        self.noise_clip = config.get("noise_clip") * self.max_action
        
        # Exploration noise added to actions when interacting with the environment
        self.expl_noise = config.get("expl_noise", 0.1)
        
        self.actor_lr = config.get("actor_lr")
        self.critic_lr = config.get("critic_lr")

        # Extract environment dimensions
        state_dim = self.state_space.shape[0]
        action_dim = self.action_space.shape[0]

        # Get architecture hyperparameters from config
        actor_hidden_sizes = config.get("actor_hidden_sizes", (64, 64))
        critic_hidden_sizes = config.get("critic_hidden_sizes", (64, 64))
        
        # Initialize the actor and critic models with the tunable architectures
        self.actor_model = TD3Actor(
            input_size=state_dim,
            hidden_sizes=actor_hidden_sizes,
            action_dim=action_dim,
            max_action=self.max_action
        ).to(device=self.device)
        
        self.critic_model = TD3Critic(
            state_dim=state_dim,
            hidden_sizes=critic_hidden_sizes,
            action_dim=action_dim
        ).to(device=self.device)

        self.trainer = TD3Trainer(
            buffer=self._replay_buffer,
            actor=self.actor_model,
            critic=self.critic_model,
            gamma=self.gamma,
            tau=self.tau,
            policy_freq=self.policy_freq,
            batch_size=self.batch_size,
            policy_noise=self.policy_noise,
            noise_clip=self.noise_clip,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            device=self.device
        )

    def add_transition(self, transition: tuple) -> None:
        """
        Add a transition to the replay buffer.
        :param transition: (state, action, reward, next_state, done)
        """
        state, action, reward, next_state, done = transition
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        action_t = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        reward_t = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
        next_state_t = torch.as_tensor(next_state, device=self.device, dtype=torch.float32)
        done_t = torch.as_tensor([float(done)], device=self.device, dtype=torch.float32)
        self._replay_buffer.push((state_t, action_t, reward_t, next_state_t, done_t))

    def update(self) -> None:
        """
        Perform a gradient descent step on both actor and critic.
        """
        return self.trainer.train()

    def policy(self, state) -> np.array:
        """
        Get the action to take based on the current state, adding Gaussian exploration noise.
        This mimics the original code:
          action = actor(state) + N(0, expl_noise * max_action)
        and clips the result between -max_action and max_action.
        """
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            action = self.actor_model(state_t.unsqueeze(0))
        action = action.cpu().numpy().flatten()
        noise = np.random.normal(0, self.expl_noise * self.max_action, size=action.shape)
        action = np.clip(action + noise, -self.max_action, self.max_action)
        return action

    def save(self, file_path='./saved_models/') -> None:
        """
        Save the actor and critic networks.
        """
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.actor_model.state_dict(), os.path.join(file_path, "td3_actor.pth"))
        torch.save(self.critic_model.state_dict(), os.path.join(file_path, "td3_critic.pth"))

    def load(self, file_path='./saved_models/') -> None:
        """
        Load the actor and critic networks, and synchronize the trainer's target networks.
        """
        self.actor_model.load_state_dict(torch.load(os.path.join(file_path, "td3_actor.pth")))
        self.critic_model.load_state_dict(torch.load(os.path.join(file_path, "td3_critic.pth")))
        self.trainer.actor_target = copy.deepcopy(self.actor_model)
        self.trainer.critic_target = copy.deepcopy(self.critic_model)
