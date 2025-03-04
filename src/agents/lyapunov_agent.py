import torch
import numpy as np

from agents.abstract_agent import AbstractAgent
from models.lyapunov_actor import LyapunovActor
from models.lyapunov_critic import LyapunovCritic
from src.trainers.lyapunov_ac_trainer import LyapunovACTrainer
from models.sampling import sample_two_headed_gaussian_model


class LyapunovACAgent(AbstractAgent):
    def __init__(self, config: dict):
        """
        We follow the same style as ActorCriticAgent, but we have a different 
        "actor" (LyapunovActor) and "critic" (LyapunovCritic) and trainer that does PDE-based learning.
        """
        super().__init__(config)
        
        self.alpha = config.get("alpha")
        self.actor_lr = config.get("actor_lr")
        self.critic_lr = config.get("critic_lr")
        
        # We assume pendulum or a single DOF environment
        state_dim = self.state_space.shape[0]
        action_dim = self.action_space.shape[0]
        max_action = 2.0   # for pendulum (action in [-2,2])
        
        # Build our Lyapunov Actor + Critic
        self._actor_model = LyapunovActor(state_dim, hidden_sizes=(64,64), 
                                          action_dim=action_dim
                                          ).to(device=self.device)
        self._critic_model = LyapunovCritic(state_dim, hidden_sizes=(64,64)
                                            ).to(device=self.device)

        dynamics_fn = config.get("dynamics_fn")
        r2_bounds = config.get("R2_bounds")

        self._trainer = LyapunovACTrainer(
            buffer=self._replay_buffer,
            actor=self._actor_model,
            critic=self._critic_model,
            dynamics_fn=dynamics_fn,
            alpha=self.alpha,
            device=self.device,
            r2_bounds=r2_bounds,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr
        )

    def add_transition(self, transition: tuple) -> None:
        state, action, reward, next_state, done = transition
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        action_t = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        reward_t = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
        next_state_t = torch.as_tensor(next_state, device=self.device, dtype=torch.float32)
        done_t = torch.as_tensor(done, device=self.device, dtype=torch.bool)
        self._replay_buffer.push((state_t, action_t, reward_t, next_state_t, done_t))

    def update(self):
        return self._trainer.train()

    def policy(self, state):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = sample_two_headed_gaussian_model(self._actor_model, s_tensor)
        return action.cpu().numpy().flatten()

    def save(self, file_path='./'):
        torch.save(self._actor_model.state_dict(), file_path + "lyapunov_actor_model.pth")
        torch.save(self._critic_model.state_dict(), file_path + "lyapunov_critic_model.pth")

    def load(self, file_path='./'):
        self._actor_model.load_state_dict(torch.load(file_path + "lyapunov_actor_model.pth"))
        self._critic_model.load_state_dict(torch.load(file_path + "lyapunov_critic_model.pth"))

    def compute_lyapunov(self, points: np.ndarray) -> np.ndarray:
        if points.shape[1] == 2:
            theta = points[:,0]
            theta_dot = points[:,1]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            obs = np.column_stack([cos_theta, sin_theta, theta_dot])
        else:
            obs = points
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            W_vals = self._critic_model(obs_t)
        return W_vals.cpu().numpy()
