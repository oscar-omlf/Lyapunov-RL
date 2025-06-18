import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR

from agents.dual_policy_agent import DualPolicyAgent
from models.lyapunov_actor import LyapunovActor
from models.lyapunov_critic import LyapunovCritic
from trainers.las_lyapunov_trainer import LAS_LyapunovAC_Trainer


class LAS_LyapunovAgent(DualPolicyAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.alpha_zubov = config.get("alpha")
        self.lr = config.get("lr")
        self.batch_size = config.get("batch_size")
        self.num_paths_sampled = config.get("num_paths_sampled")
        self.norm_threshold = config.get("norm_threshold")
        self.integ_threshold = config.get("integ_threshold")
        self.dt = config.get("dt")
        self.r1_bounds = config.get("r1_bounds")
        
        actor_hidden_sizes = config.get("actor_hidden_sizes")
        critic_hidden_sizes = config.get("critic_hidden_sizes")
        
        self.actor_model = LyapunovActor(self.state_dim, actor_hidden_sizes, self.action_dim, max_action=self.max_action).to(device=self.device)
        self.critic_model = LyapunovCritic(self.state_dim, critic_hidden_sizes).to(device=self.device)

        self.optimizer = torch.optim.Adam(list(self.actor_model.parameters()) + list(self.critic_model.parameters()), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.8)
        
        self.run_dir = config.get("run_dir")

        L_inv = torch.linalg.inv(torch.as_tensor(self.lqr_agent.P_np, dtype=torch.float32))
        self.L_inv = L_inv

        self.trainer = LAS_LyapunovAC_Trainer(
            agent=self,
            alpha_zubov=self.alpha_zubov,
            batch_size=self.batch_size,
            num_paths_sampled=self.num_paths_sampled,
            norm_threshold=self.norm_threshold,
            integ_threshold=self.integ_threshold,
            dt=self.dt,
            dynamics_fn=self.dynamics_fn,
            dynamics_fn_dreal=self.dynamics_fn_dreal,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            r1_bounds=self.r1_bounds,
            run_dir=self.run_dir,
            device=self.device,
        )

    def _get_global_action(self, state_torch: torch.Tensor) -> torch.Tensor:
        mu_theta_action = self.actor_model(state_torch)
        return mu_theta_action
    
    def get_composite_W_value(self, state_torch: torch.Tensor) -> torch.Tensor:
        W_glo = self.critic_model(state_torch)
        if W_glo.ndim == 1:
            W_glo = W_glo.unsqueeze(-1)

        V_loc = self.blending_function.get_lyapunov_value(state_torch)
        W_loc = torch.tanh(self.alpha_zubov * V_loc)
        if W_loc.ndim == 1:
            W_loc = W_loc.unsqueeze(-1)

        h2_val = self.blending_function.get_h2(state_torch)
        if h2_val.ndim == 1:
            h2_val = h2_val.unsqueeze(-1)

        W_comp_val = W_loc + h2_val * (W_glo - W_loc)
        return W_comp_val.squeeze(-1)

    def add_transition(self, transition: tuple) -> None:
        pass

    def update(self, counter_examples: list = None, normalize_gradients: bool = False):
        loss = self.trainer.train(counter_examples=counter_examples, normalize_gradients=normalize_gradients)
        return loss
    
    def save(self, file_path, episode=None):
        torch.save(self.actor_model.state_dict(), file_path + "actor_model_" + str(episode) + ".pth")
        torch.save(self.critic_model.state_dict(), file_path + "critic_model_" + str(episode) + ".pth")

    def load(self, file_path, episode=None):
        self.actor_model.load_state_dict(torch.load(file_path + 'actor_model_' + str(episode) + '.pth'))
        self.critic_model.load_state_dict(torch.load(file_path + 'critic_model_' + str(episode) + '.pth'))
    