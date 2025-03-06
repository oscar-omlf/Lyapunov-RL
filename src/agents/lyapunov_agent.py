import torch
import numpy as np

from agents.abstract_agent import AbstractAgent
from models.lyapunov_actor import LyapunovActor
from models.lyapunov_critic import LyapunovCritic
from trainers.lyapunov_ac_trainer import LyapunovACTrainer
from util.sampling import sample_two_headed_gaussian_model


class LyapunovACAgent(AbstractAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.alpha = config.get("alpha")
        self.actor_lr = config.get("actor_lr")
        self.critic_lr = config.get("critic_lr")
        dynamics_fn = config.get("dynamics_fn")
        self.dt = config.get("dt")
        self.norm_threshold = config.get("norm_threshold")
        r1_bounds = config.get("r1_bounds")
        
        state_dim = self.state_space.shape[0]
        action_dim = self.action_space.shape[0]
        
        self._actor_model = LyapunovActor(state_dim, hidden_sizes=(64,64), action_dim=action_dim).to(device=self.device)
        self._critic_model = LyapunovCritic(state_dim, hidden_sizes=(64,64)).to(device=self.device)


        self._trainer = LyapunovACTrainer(
            actor=self._actor_model,
            critic=self._critic_model,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            alpha=self.alpha,
            batch_size=config.get("batch_size"),
            num_paths_sampled=config.get("num_paths_sampled"),
            norm_threshold=self.norm_threshold,
            dt=self.dt,
            dynamics_fn=dynamics_fn,
            state_space=state_dim,
            r1_bounds=r1_bounds,
            device=self.device
        )

    def add_transition(self, transition: tuple) -> None:
        pass

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
        """
        Compute the Lyapunov function values W(x) using the critic network.
        If points have 2 columns, they are assumed to be [theta, theta_dot] and are converted
        to the 3D observation [cos(theta), sin(theta), theta_dot]. Otherwise, points are used as is.
        This function is used to estimate the Domain of Attraction (DoA).
        """
        if points.shape[1] == 2:
            theta = points[:, 0]
            theta_dot = points[:, 1]
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            obs = np.column_stack([cos_theta, sin_theta, theta_dot])
        else:
            obs = points
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            W_vals = self._critic_model(obs_t)
        return W_vals.cpu().numpy()
