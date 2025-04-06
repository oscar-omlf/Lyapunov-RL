import numpy as np
import torch
from typing import Callable
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt

from util.sampling import sample_in_region_gpu, sample_out_of_region_gpu
from util.rk4_step import rk4_step
from trainers.abstract_trainer import Trainer


class LyapunovACTrainer(Trainer):
    def __init__(
        self, 
        actor: nn.Module, 
        critic: nn.Module,
        actor_lr: float, 
        critic_lr: float,
        alpha: float,
        batch_size: int,
        num_paths_sampled: int,
        norm_threshold: float,
        integ_threshold: int,
        dt: float,
        dynamics_fn: Callable,
        state_space: int,
        r1_bounds: list,
        device: str
    ):
        super().__init__()
        self.actor_model = actor
        self.critic_model = critic

        self.alpha = alpha
        self.batch_size = batch_size
        self.num_paths_sampled = num_paths_sampled
        self.norm_threshold = norm_threshold
        self.integ_threshold = integ_threshold
        self.dt = dt

        self.dynamics_fn = dynamics_fn

        self.state_space = state_space
        self.lb, self.ub = r1_bounds

        self.device = device

        self.lb_tensor = torch.tensor(self.lb, dtype=torch.float32, device=self.device)
        self.ub_tensor = torch.tensor(self.ub, dtype=torch.float32, device=self.device)

        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=500, gamma=0.8)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=500, gamma=0.8)

        self.timesteps = 0

        print('Lyapunov Trainer Initialized!')

    def train(self):
        # Use GPU-based sampling and vectorized trajectory simulation
        init_states = sample_in_region_gpu(self.num_paths_sampled, self.lb_tensor, self.ub_tensor, self.device)
        traj, values, conv_flags = self.simulate_trajectories(init_states, max_steps=3000)
        # Here, 'values' is assumed to be the integrated norm computed during simulation.
        values = values.to(dtype=torch.float32, device=self.device)

        # 1) Enforce W(0) = 0
        zeros_tensor = torch.zeros((1, self.state_space), dtype=torch.float32, device=self.device)
        Lz = 5 * torch.square(self.critic_model(zeros_tensor))

        # 2) Enforce W(x) = tanh(alpha * V(x))
        Wx = self.critic_model(init_states)
        target = torch.tanh(self.alpha * values)
        Lr = F.mse_loss(Wx, target)

        # 3) Physics-Informed Loss (PDE residual)
        init_states_in = sample_in_region_gpu(self.batch_size, self.lb_tensor, self.ub_tensor, self.device)
        Wx_in, grad_Wx = self.critic_model.forward_with_grad(init_states_in)
        us = self.actor_model(init_states_in)
        fxu = self.dynamics_fn(init_states_in, us)
        phix = torch.linalg.vector_norm(init_states_in, ord=2, dim=1)
        resid = torch.sum(grad_Wx * fxu.detach(), dim=1) + self.alpha * (1 + Wx_in.squeeze()) * (1 - Wx_in.squeeze()) * phix
        Lp = torch.mean(torch.square(resid))

        # 4) Encourage control actions that decrease the Lyapunov function
        grad_norm = torch.linalg.vector_norm(grad_Wx, ord=2, dim=1, keepdim=True)
        unit_grad = grad_Wx / (grad_norm + 1e-8)
        Lc = torch.mean(torch.sum(unit_grad.detach() * fxu, dim=1))

        # 5) Enforce that on the boundary of R2, W(x) ≈ 1
        init_states_out = sample_out_of_region_gpu(self.batch_size, self.lb_tensor, self.ub_tensor, scale=2, device=self.device)
        Wx_out = self.critic_model(init_states_out)
        Lb = 2 * torch.mean(torch.abs(Wx_out - 1.0))

        actor_loss = Lc
        critic_loss = Lz + Lr + Lp + Lb

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_scheduler.step()
        self.critic_scheduler.step()

        self.timesteps += 1
        if self.timesteps % 20 == 0:
            self.plot_level_set_and_trajectories()

        return actor_loss.item(), critic_loss.item()

    @torch.no_grad()
    def simulate_trajectories(self, x: torch.Tensor, max_steps: int = 3000):
        """
        Vectorized simulation for a batch of trajectories.
        :param x: Initial state tensor of shape [B, state_space].
        :param max_steps: Maximum simulation steps.
        :return: traj: tensor of shape [B, T, state_space] containing the full trajectory history,
                integ_acc: integrated norm per trajectory,
                converged: convergence flags.
        """
        B = x.shape[0]
        integ_acc = torch.zeros(B, device=self.device)
        active = torch.ones(B, dtype=torch.bool, device=self.device)
        x_hist = [x.clone()]
        buffer = x.clone()

        for step in range(max_steps):
            if not active.any():
                break

            norm = torch.linalg.vector_norm(x, ord=2, dim=1)
            integ_acc[active] += norm[active] * self.dt

            converged = norm < self.norm_threshold
            if step >= 10:
                stabilization = torch.linalg.vector_norm(x - buffer, ord=2, dim=1) < 1e-3
            else:
                stabilization = torch.zeros_like(norm, dtype=torch.bool)
            diverged = integ_acc > self.integ_threshold
            finished = converged | stabilization | diverged
            active = active & (~finished)

            if step % 10 == 0:
                buffer = x.clone()

            if active.any():
                u = self.actor_model(x)
                if u.dim() == 1:
                    u = u.unsqueeze(1)
                x_next = rk4_step(self.dynamics_fn, x, u, dt=self.dt)
                # Only update the active trajectories
                x = torch.where(active.unsqueeze(1), x_next, x)
                x_hist.append(x.clone())

        # Stack the history along a new time dimension: shape [B, T, state_space]
        traj = torch.stack(x_hist, dim=1)
        final_norm = torch.linalg.vector_norm(x, ord=2, dim=1)
        converged = final_norm < self.norm_threshold

        return traj, integ_acc, converged

    def plot_level_set_and_trajectories(self):
        # Create a grid covering R2 (boundary scaled by 2)
        x_min, x_max = self.lb[0]*2, self.ub[0]*2
        y_min, y_max = self.lb[1]*2, self.ub[1]*2

        print(x_min, x_max, y_min, y_max)
        # Use 3000 points and 100 contour levels as in the original
        xx = np.linspace(x_min, x_max, 3000)
        yy = np.linspace(y_min, y_max, 3000)
        X, Y = np.meshgrid(xx, yy)
        grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
        grid_tensor = torch.as_tensor(grid_points, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            Z = self.critic_model(grid_tensor).cpu().numpy()
        Z = Z.reshape(X.shape)
        
        plt.figure(figsize=(6,5))
        cp = plt.contourf(X, Y, Z, levels=100, cmap='viridis')
        plt.colorbar(cp)
        plt.title("Critic Level Set (Lyapunov Function)")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        # Explicitly set axis limits to match the grid boundaries
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # Vectorized simulation: simulate 5 trajectories in parallel
        init_states = sample_in_region_gpu(5, self.lb_tensor, self.ub_tensor, self.device)
        trajs, _, conv_flags = self.simulate_trajectories(init_states, max_steps=3000)
        trajs_np = trajs.cpu().numpy()
        
        # Plot each trajectory
        for i in range(trajs_np.shape[0]):
            plt.plot(trajs_np[i, :, 0], trajs_np[i, :, 1], 'o-', markersize=1, linewidth=0.5, color='r')
            plt.plot(trajs_np[i, 0, 0], trajs_np[i, 0, 1], 'r+')
        
        plt.gca().set_aspect('equal')
        plt.savefig(f"./plots/lyAC_{self.timesteps}.png")
        plt.close()
