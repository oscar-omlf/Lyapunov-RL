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
        self.dynamics_fn = config.get("dynamics_fn")
        self.batch_size = config.get("batch_size")
        self.num_paths_sampled = config.get("num_paths_sampled")
        self.dt = config.get("dt")
        self.norm_threshold = config.get("norm_threshold")
        self.integ_threshold = config.get("integ_threshold")
        self.r1_bounds = config.get("r1_bounds")
        
        state_dim = self.state_space
        action_dim = self.action_space

        actor_hidden_sizes = config.get("actor_hidden_sizes", (5, 5))
        critic_hidden_sizes = config.get("critic_hidden_sizes", (20, 20))

        self.max_action = 2.0
        
        self._actor_model = LyapunovActor(state_dim, actor_hidden_sizes, action_dim, max_action=self.max_action).to(device=self.device)
        self._critic_model = LyapunovCritic(state_dim, critic_hidden_sizes).to(device=self.device)


        self._trainer = LyapunovACTrainer(
            actor=self._actor_model,
            critic=self._critic_model,
            actor_lr=self.actor_lr,
            critic_lr=self.critic_lr,
            alpha=self.alpha,
            batch_size=self.batch_size,
            num_paths_sampled=self.num_paths_sampled,
            norm_threshold=self.norm_threshold,
            integ_threshold=self.integ_threshold,
            dt=self.dt,
            dynamics_fn=self.dynamics_fn,
            state_space=state_dim,
            r1_bounds=self.r1_bounds,
            device=self.device
        )

    def add_transition(self, transition: tuple) -> None:
        pass

    def update(self):
        return self._trainer.train()

    def policy(self, state):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self._actor_model(s_tensor)
        return action.cpu().numpy().flatten()

    def save(self, file_path='./saved_models/'):
        torch.save(self._actor_model.state_dict(), file_path + "lyapunov_actor_model.pth")
        torch.save(self._critic_model.state_dict(), file_path + "lyapunov_critic_model.pth")

    def load(self, file_path='./saved_models/'):
        self._actor_model.load_state_dict(torch.load(file_path + "lyapunov_actor_model.pth"))
        self._critic_model.load_state_dict(torch.load(file_path + "lyapunov_critic_model.pth"))

    def compute_lyapunov(self, points: np.ndarray) -> np.ndarray:
        """
        Compute the Lyapunov function values W(x) using the critic network.
        This function is used to estimate the Domain of Attraction (DoA).
        """
        obs_t = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            W_vals = self._critic_model(obs_t)
        return W_vals.cpu().numpy()
