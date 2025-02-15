import torch
import torch.nn as nn
from agents.abstract_agent import AbstractAgent
from models.actor import NStepActor
from models.critic import NStepCritic
from trainers.ac_trainer import ACTrainer

class ActorCriticAgent(AbstractAgent):
    def __init__(self, config):
        """
        Initialize the n-step actor-critic agent.
        """
        super().__init__(config)
        
        # Extract hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.n_steps = config.get("n_steps", 3)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.actor_lr = config.get("actor_lr", 0.0001)
        self.critic_lr = config.get("critic_lr", 0.0005)
        self.device = config.get("device", "cpu")
        
        # Extract the environment dimensions
        state_dim = config["state_space"].shape[0]
        action_dim = config["action_space"].shape[0] # Assumes continuous action space (like Pendulum-v1)!!
        
        # Initialize the actor and critic models
        self.actor = NStepActor(state_dim, action_dim).to(self.device)
        self.critic = NStepCritic(state_dim).to(self.device)
        
        # Initialize the trainer that handles training the actor-critic
        self.trainer = ACTrainer(
            buffer=self.replay_buffer,
            actor=self.actor,
            critic=self.critic,
            gamma=self.gamma,
            n_steps=self.n_steps,
            entropy_coef=self.entropy_coef,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            device=self.device
        )
    
    def policy(self, state):
        """
        Given a state, use the actor network to sample an action.
        The state is converted to a torch tensor, passed to the actor's predict() method,
        and an action is sampled from the resulting MultivariateNormal distribution.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.actor.predict(state_tensor)
            action = dist.sample()
        return action.cpu().numpy()[0]
    
    def update(self):
        """
        Update the actor and critic parameters.
        The ACTrainer handles the actual training, so this method simply calls train().
        """
        return self.trainer.train()
    
    def save(self, filepath):
        """
        Save the actor and critic network parameters to separate files.
        """
        torch.save(self.actor.state_dict(), filepath + "_actor.pth")
        torch.save(self.critic.state_dict(), filepath + "_critic.pth")
    
    def load(self, filepath):
        """
        Load the actor and critic network parameters from the specified files.
        """
        self.actor.load_state_dict(torch.load(filepath + "_actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(filepath + "_critic.pth", map_location=self.device))
