import os
import copy
import torch
import numpy as np

from agents.dual_policy_agent import DualPolicyAgent
from models.td3_actor import TD3Actor
from models.td3_critic import TD3Critic
from trainers.las_td3_trainer import LAS_TD3Trainer


class LAS_TD3Agent(DualPolicyAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.gamma = config.get("gamma")
        self.tau = config.get("tau")
        self.policy_freq = config.get("policy_freq")
        self.batch_size = config.get("batch_size")        
        self.policy_noise = config.get("policy_noise") * self.max_action
        self.noise_clip = config.get("noise_clip") * self.max_action
        self.expl_noise = config.get("expl_noise")
        
        self.actor_lr = config.get("actor_lr")
        self.critic_lr = config.get("critic_lr")

        # Get architecture hyperparameters from config
        actor_hidden_sizes = config.get("actor_hidden_sizes", (64, 64))
        critic_hidden_sizes = config.get("critic_hidden_sizes", (64, 64))
        
        # Initialize the actor and critic models with the tunable architectures
        self.actor_model = TD3Actor(
            input_size=self.state_dim,
            hidden_sizes=actor_hidden_sizes,
            action_dim=self.action_dim,
            max_action=self.max_action
        ).to(device=self.device)
        
        self.critic_model = TD3Critic(
            state_dim=self.state_dim,
            hidden_sizes=critic_hidden_sizes,
            action_dim=self.action_dim
        ).to(device=self.device)

        # Target networks
        self.actor_target = copy.deepcopy(self.actor_model)
        self.critic_target = copy.deepcopy(self.critic_model)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)

        H_matrix, H_matrix_np = self._initialize_H_matrix()
        self.H_matrix = H_matrix
        self.H_matrix_np = H_matrix_np

        self.trainer = LAS_TD3Trainer(
            agent=self,
            buffer=self._replay_buffer,
            gamma=self.gamma,
            tau=self.tau,
            policy_freq=self.policy_freq,
            batch_size=self.batch_size,
            policy_noise=self.policy_noise,
            noise_clip=self.noise_clip,
            device=self.device
        )

    def _initialize_H_matrix(self):
        # Compute and store the H matrix for Q_loc
        if not self.lqr_agent.discrete_discounted:
            raise ValueError("TD3 requires a discrete-time LQR agent.")

        A = self.lqr_agent.A_np
        B = self.lqr_agent.B_np

        P = self.lqr_agent.P_np

        lqr_gamma = self.lqr_agent.gamma
        
        Q_gamma = self.lqr_agent.Q_gamma
        R_gamma = self.lqr_agent.R_gamma

        H_matrix_np = self.riccati_solver.compute_H_matrix(
            A, B, Q_gamma, R_gamma, P, lqr_gamma
        )

        H_matrix = torch.from_numpy(H_matrix_np).to(dtype=torch.float32, device=self.device)

        return H_matrix, H_matrix_np

    def _compute_Q_loc(self, state_torch: torch.Tensor, action_torch: torch.Tensor) -> torch.Tensor:
        current_x_star = self.x_star.unsqueeze(0)
        current_u_star = self.u_star.unsqueeze(0)

        delta_x = state_torch - current_x_star
        delta_u = action_torch - current_u_star

        if delta_u.shape[0] != delta_x.shape[0] and delta_u.shape[0] == 1:
            delta_u = delta_u.expand(delta_x.shape[0], -1)


        z_error = torch.cat((delta_x, delta_u), dim=-1)

        term1 = torch.matmul(z_error, self.H_matrix)
        Q_loc_val = torch.sum(term1 * z_error, dim=1, keepdim=True)
        return Q_loc_val
    
    def get_composite_Q_values(
            self, 
            state_torch: torch.Tensor,
            action_torch: torch.Tensor,
            use_target_critic: bool = False    
        ):
        critic_net_to_use = self.critic_target if use_target_critic else self.critic_model
        omega_q1, omega_q2 = critic_net_to_use(state_torch, action_torch)

        # return omega_q1, omega_q2

        q_loc = self._compute_Q_loc(state_torch, action_torch)

        with torch.no_grad():
            h2_val = self.blending_function.get_h2(state_torch)
            if h2_val.ndim == 1:
                h2_val = h2_val.unsqueeze(-1)

        Q_comp1 = q_loc + h2_val * (omega_q1 - q_loc)
        Q_comp2 = q_loc + h2_val * (omega_q2 - q_loc)
        return Q_comp1, Q_comp2

    def get_composite_Q1_value(
        self,
        state_torch: torch.Tensor,
        action_torch: torch.Tensor,
    ):
        omega_q1 = self.critic_model.Q1_value(state_torch, action_torch)
        # return omega_q1

        q_loc = self._compute_Q_loc(state_torch, action_torch)

        with torch.no_grad():
            h2_val = self.blending_function.get_h2(state_torch)
            if h2_val.ndim == 1:
                h2_val = h2_val.unsqueeze(-1)

        Q_comp1 = q_loc + h2_val * (omega_q1 - q_loc)
        return Q_comp1

    def _get_global_action(self, state_torch: torch.Tensor, noise: bool = True) -> torch.Tensor:
        mu_theta_action = self.actor_model(state_torch)
        

        if self.expl_noise > 0 and noise is True:
            noise = torch.normal(
                0, self.expl_noise * self.max_action,
                size=mu_theta_action.shape,
                device=self.device
            )
            mu_theta_action += noise

        mu_theta_action = torch.clamp(mu_theta_action, -self.max_action, self.max_action)
        return mu_theta_action

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

    def save(self, file_path, episode=None) -> None:
        """
        Save the actor and critic networks.
        """
        os.makedirs(file_path, exist_ok=True)
        torch.save(self.actor_model.state_dict(), os.path.join(file_path, "td3_actor_" + str(episode) + ".pth"))
        torch.save(self.critic_model.state_dict(), os.path.join(file_path, "td3_critic_" + str(episode) + ".pth"))

    def load(self, file_path, episode=None) -> None:
        """
        Load the actor and critic networks, and synchronize the trainer's target networks.
        """
        self.actor_model.load_state_dict(torch.load(os.path.join(file_path, "td3_actor_" + str(episode) + ".pth"), map_location=torch.device(self.device)))
        self.critic_model.load_state_dict(torch.load(os.path.join(file_path, "td3_critic_" + str(episode) + ".pth"), map_location=torch.device(self.device)))
        self.actor_target = copy.deepcopy(self.actor_model)
        self.critic_target = copy.deepcopy(self.critic_model)
