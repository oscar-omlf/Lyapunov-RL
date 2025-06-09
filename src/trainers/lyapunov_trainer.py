import os
from typing import Callable
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Union

from agents.abstract_agent import AbstractAgent
from trainers.abstract_trainer import Trainer
from util.sampling import sample_in_region_torch, sample_out_of_region_torch
from util.rk4_step import rk4_step

import dreal as d
from util.dreal import dreal_var, in_box, on_boundary, is_unsat


class LyapunovTrainer(Trainer):
    def __init__(
        self, 
        actor: nn.Module, 
        critic: nn.Module,
        lr: float, 
        alpha: float,
        batch_size: int,
        num_paths_sampled: int,
        norm_threshold: float,
        integ_threshold: int,
        dt: float,
        dynamics_fn: Callable,
        dynamics_fn_dreal: Callable,
        state_dim: int,
        r1_bounds: list,
        run_dir: str,
        device: str,
    ):
        super().__init__()
        self.actor_model = actor
        self.critic_model = critic

        self.alpha_zubov = alpha
        self.batch_size = batch_size
        self.num_paths_sampled = num_paths_sampled
        self.norm_threshold = norm_threshold
        self.integ_threshold = integ_threshold
        self.dt = dt

        self.dynamics_fn = dynamics_fn
        self.dynamics_fn_dreal = dynamics_fn_dreal

        self.state_dim = state_dim
        self.lb, self.ub = r1_bounds

        self.run_dir = run_dir

        self.device = device

        self.lb_tensor = torch.tensor(self.lb, dtype=torch.float32, device=self.device)
        self.ub_tensor = torch.tensor(self.ub, dtype=torch.float32, device=self.device)

        self.optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.8)

        self.timesteps = 0

        print('Lyapunov Trainer Initialized (Standalone LAC)!')

    def train(self, counter_examples: list = None):
        # Use GPU-based sampling and vectorized trajectory simulation
        init_states = sample_in_region_torch(self.num_paths_sampled, self.lb_tensor, self.ub_tensor, self.device)
        traj, values, _ = self.simulate_trajectories(init_states, max_steps=3000)
        values = values.to(dtype=torch.float32, device=self.device)

        # 1) Enforce W(0) = 0
        zeros_tensor = torch.zeros((1, self.state_dim), dtype=torch.float32, device=self.device)
        W_zeros = self.critic_model(zeros_tensor)
        Lz = 5.0 * torch.square(W_zeros) 

        # 2) Enforce W(x) = tanh(alpha * V(x))
        Wx_Lr = self.critic_model(init_states)
        target = torch.tanh(self.alpha_zubov * values)
        Lr = F.mse_loss(Wx_Lr, target)

        # 3) Physics-Informed Loss (PDE residual)
        if counter_examples is not None and len(counter_examples) > 0:
            ce_tensor = torch.as_tensor(counter_examples, dtype=torch.float32, device=self.device)
            random_samples = sample_in_region_torch(self.batch_size - len(counter_examples), self.lb_tensor, self.ub_tensor, self.device)
            init_states_in = torch.cat([ce_tensor, random_samples], dim=0)
        else:
            init_states_in = sample_in_region_torch(self.batch_size, self.lb_tensor, self.ub_tensor, self.device)

        Wx_in, grad_Wx_in = self.critic_model.forward_with_grad(init_states_in)

        current_actions = self.actor_model(init_states_in)
        current_fxu = self.dynamics_fn(init_states_in, current_actions)
        
        phix = torch.norm(init_states_in, p=2, dim=1) 

        # changed current_fxu to fxu_detached
        resid = torch.sum(grad_Wx_in * current_fxu.detach(), dim=1) + \
        self.alpha_zubov * (1 + Wx_in.squeeze()) * (1 - Wx_in.squeeze()) * phix
        Lp = torch.mean(torch.square(resid))

        # 4) Encourage control actions that decrease the Lyapunov function
        # grad_norm = torch.linalg.vector_norm(grad_Wx_in, ord=2, dim=1, keepdim=True)
        # unit_grad = grad_Wx_in / (grad_norm + 1e-8)
        # Lc = 0.5 *torch.mean(torch.sum(unit_grad.detach() * current_fxu, dim=1))
        Lc = 0.5 * torch.mean(torch.sum(grad_Wx_in.detach() * current_fxu, dim=1))

        # 5) Enforce that on the boundary of R2, W(x) ≈ 1
        init_states_out = sample_out_of_region_torch(self.batch_size, self.lb_tensor, self.ub_tensor, scale=2, device=self.device)
        Wx_out = self.critic_model(init_states_out) 
        # Lb = 5.0 * torch.mean(torch.abs(Wx_out - 1.0))
        Lb = 5.0 * F.l1_loss(Wx_out, torch.ones_like(Wx_out).to(self.device))

        actor_loss = Lc
        critic_loss = Lz + Lr + Lp + Lb

        total_loss = 0.5 * actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.timesteps += 1
        if self.timesteps % 20 == 0:
            self.plot_level_set_and_trajectories()

        print(f"Lz: {Lz.item():.4f} | Lr: {Lr.item():.4f} | Lp: {Lp.item():.4f} | Lc: {Lc.item():.4f} | Lb: {Lb.item():.4f}")
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

        # Use 3000 points and 100 contour levels as in the original
        xx = np.linspace(x_min, x_max, 3000)
        yy = np.linspace(y_min, y_max, 3000)
        X, Y = np.meshgrid(xx, yy)
        grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
        grid_tensor = torch.as_tensor(grid_points, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            Z = self.critic_model(grid_tensor)
            Z = Z.cpu().numpy()
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
        init_states = sample_in_region_torch(5, self.lb_tensor, self.ub_tensor, self.device)
        trajs, _, _ = self.simulate_trajectories(init_states, max_steps=3000)
        trajs_np = trajs.cpu().numpy()
        
        # Plot each trajectory
        for i in range(trajs_np.shape[0]):
            plt.plot(trajs_np[i, :, 0], trajs_np[i, :, 1], 'o-', markersize=1, linewidth=0.5, color='r')
            plt.plot(trajs_np[i, 0, 0], trajs_np[i, 0, 1], 'r+')
        
        os.makedirs("plots", exist_ok=True)

        plt.gca().set_aspect('equal')
        level_set_dir = os.path.join(self.run_dir, "level_sets")
        os.makedirs(level_set_dir, exist_ok=True)
        plot_path = os.path.join(level_set_dir, f"level_set_{self.timesteps}.png")
        plt.savefig(plot_path)
        plt.close()

    def _sanity_network(self):
        x_np = np.random.randn(self.state_dim)
        x_d  = dreal_var(self.state_dim)
        y_pt = self.actor_model(torch.as_tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0))[0].item()
        y_dr = self.actor_model.forward_dreal(x_d)[0]
        fsat = d.And(*[x_d[i] == x_np[i] for i in range(self.state_dim)],
                     y_dr >= y_pt - 1e-3,
                     y_dr <= y_pt + 1e-3)
        assert d.CheckSatisfiability(fsat, 1e-4) is not None

    def _sanity_dynamics(self):
        x_np = np.random.randn(self.state_dim)
        u_np = np.random.randn(1)
        x_d  = dreal_var(self.state_dim)
        u_d  = dreal_var(1, "u")
        f_pt = self.dynamics_fn(torch.as_tensor(x_np).unsqueeze(0),
                                torch.as_tensor(u_np).unsqueeze(0))[0].cpu().numpy()
        f_dr = self.dynamics_fn_dreal(x_d, u_d)
        cons = [x_d[i] == x_np[i] for i in range(self.state_dim)] + \
               [u_d[0] == u_np[0]] + \
               [f_dr[i] >= f_pt[i]-1e-3 for i in range(self.state_dim)] + \
               [f_dr[i] <= f_pt[i]+1e-3 for i in range(self.state_dim)]
        assert d.CheckSatisfiability(d.And(*cons), 1e-4) is not None

    def in_domain_dreal(self, x, scale=1.0):
        return d.And(
            x[0] >= self.lb[0] * scale,
            x[0] <= self.ub[0] * scale,
            x[1] >= self.lb[1] * scale,
            x[1] <= self.ub[1] * scale
        )
    
    def on_boundary_dreal(self, x, scale=2.0):
        condition1 = d.And(
            x[0] >= self.lb[0] * scale * 0.99,
            x[0] <= self.ub[0] * scale * 0.99,
            x[1] >= self.lb[1] * scale * 0.99,
            x[1] <= self.ub[1] * scale * 0.99
        )
        condition2 = d.Not(
            d.And(
                x[0] >= self.lb[0] * scale * 0.97,
                x[0] <= self.ub[0] * scale * 0.97,
                x[1] >= self.lb[1] * scale * 0.97,
                x[1] <= self.ub[1] * scale * 0.97
            )
        )
        return d.And( condition1, condition2 )

    def check_lyapunov(self, level=0.9, scale=2., eps=0.5):
        print('Standalone LyAC Lyapunov Checker')
        W0 = self.critic_model(torch.zeros((1, self.state_dim), device=self.device))
        W0 = W0.squeeze().item()
        x = dreal_var(self.state_dim)
        x_norm = d.Expression(0.)
        lie_derivative_W = d.Expression(0.)

        # construct xnorm and f(x, u)^T \nabla_x W(x)
        u = self.actor_model.forward_dreal(x)
        fx = self.dynamics_fn_dreal(x, u)
        Wx = self.critic_model.forward_dreal(x)[0]

        # construct x_norm and <fx, \grad_x W(x)>
        for i in range(self.state_dim):
            x_norm += x[i] * x[i]
            lie_derivative_W += fx[i] * Wx.Differentiate(x[i])
        x_norm = d.sqrt(x_norm)

        condition = d.And(
            x_norm >= eps,
            self.in_domain_dreal(x, scale),
            Wx <= level,
            d.Or(
                lie_derivative_W >= 0,
                Wx <= W0
            )
        )
        r1 = d.CheckSatisfiability( condition, 0.01 )
        
        r2 = d.CheckSatisfiability(
            d.And(
                self.on_boundary_dreal(x, scale=scale),
                Wx <= level
            ),
            0.01
        )
        print('----------')
        print(r1)
        print(r2)
        print('----------')
        return r1, r2

    def check_lyapunov_with_ce(self, level=0.9, scale=2., eps=0.5):
        print(f"Verifying with c = {level:.4f} and eps = {eps:.2f}...")
        # Get W(0) value
        W0 = self.critic_model(torch.zeros((1, self.state_dim), device=self.device)).squeeze().item()
        
        # Define dReal variables
        x = dreal_var(self.state_dim)
        
        # Construct dReal expressions for dynamics and Lyapunov function
        u = self.actor_model.forward_dreal(x)
        fx = self.dynamics_fn_dreal(x, u)
        Wx = self.critic_model.forward_dreal(x)[0]
        
        # Construct Lie derivative and state norm
        lie_derivative_W = sum(fx[i] * Wx.Differentiate(x[i]) for i in range(self.state_dim))
        x_norm = d.sqrt(sum(xi * xi for xi in x)) # Use sqrt for direct comparison with eps

        # --- Condition 1: Find a state inside the RoA that violates conditions ---
        # This is the primary falsification query.
        # It looks for a point where:
        #   1. The norm is NOT insignificant (>= eps)
        #   2. It's within the region of interest (R2)
        #   3. The Lyapunov candidate W(x) is below the level c
        #   4. EITHER the Lie derivative is non-negative OR W(x) is not greater than W(0)
        violation_condition = d.And(
            x_norm >= eps,
            self.in_domain_dreal(x, scale), # Use the correct helper function
            Wx <= level,
            d.Or(
                lie_derivative_W >= 0,
                Wx <= W0
            )
        )
        r1 = d.CheckSatisfiability(violation_condition, 0.001)
        if r1:
            print("FALSIFIED: Found counter-example violating Lie derivative or positivity.")
            return (False, r1)

        # --- Condition 2: Check if the level set escapes the boundary of R2 ---
        boundary_condition = d.And(
            self.on_boundary_dreal(x, scale=scale),
            Wx <= level
        )
        r2 = d.CheckSatisfiability(boundary_condition, 0.001)
        if r2:
            print("FALSIFIED: RoA level set touches the boundary of the verified region.")
            return (False, r2)

        # If both checks are UNSAT, the system is verified for this level
        print("Verification PASSED for this level")
        return (True, None)
