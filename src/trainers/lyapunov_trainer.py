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


class LyapunovTrainer(Trainer):
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
        state_dim: int,
        r1_bounds: list,
        device: str,
        dual_controller_components: Union[dict, bool]
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

        self.state_dim = state_dim
        self.lb, self.ub = r1_bounds

        self.device = device

        self.lb_tensor = torch.tensor(self.lb, dtype=torch.float32, device=self.device)
        self.ub_tensor = torch.tensor(self.ub, dtype=torch.float32, device=self.device)

        # self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        # self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

        # self.actor_scheduler = StepLR(self.actor_optimizer, step_size=500, gamma=0.8)
        # self.critic_scheduler = StepLR(self.critic_optimizer, step_size=500, gamma=0.8)

        self.optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=actor_lr)
        self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.8)

        self.timesteps = 0
        self.dual_controller_components = dual_controller_components

        # If we are doing dual-policy LAS-GLOBAL
        if self.dual_controller_components:
            self.lqr_agent = dual_controller_components["LQR"]
            self.blending_function = dual_controller_components["BLENDING_FUNCTION"]
            print('Lyapunov Trainer Initialized with Dual Controller (LQR + Blending)!')
        else:
            print('Lyapunov Trainer Initialized (Standalone LyAC)!')

    def _get_current_policy_actions(self, state):
        """
        Helper to get actions from the current policy (either learnt or composite).
        """
        actions_learned = self.actor_model(state)

        if isinstance(self.dual_controller_components, dict):
            actions_loc = self.lqr_agent.policy(state)

            h1_blend = self.blending_function.get_h1(state)

            # TODO: Remove this check
            if h1_blend.ndim == 1:
                print('h1 is a scalar, not a tensor. This is probably a bug.')
                h1_blend = h1_blend.unsqueeze(-1)

            current_actions = actions_loc + h1_blend * (actions_learned - actions_loc)
        else:
            current_actions = actions_learned

        return current_actions


    def lyapunov_value(self, state: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        """
        Computes the Lyapunov/ Zubov value W(x).
        If dual_controller is active, computes W_composite(x).
        Returns W(x) and optionally its gradient w.r.t. state.
        """
        if requires_grad and not state.requires_grad:
            state.requires_grad_(True)

        Wx_learned = self.critic_model(state)

        if self.dual_controller_components:
            V_loc = self.blending_function.get_lyapunov_value(state)
            W_loc = torch.tanh(self.alpha_zubov * V_loc)

            h2 = self.blending_function.get_h2(state)
            Wx_final = W_loc + h2 * (Wx_learned - W_loc)
        else:
            Wx_final = Wx_learned

        grad_Wx = None
        if requires_grad:
            if Wx_final.requires_grad:
                grad_Wx = torch.autograd.grad(Wx_final.sum(), state, create_graph=True, retain_graph=True)[0] 
            else:
                print('Wx_final does not require grad. Setting grad_Wx to zero.')
                grad_Wx = torch.zeros_like(Wx_final)

        return Wx_final, grad_Wx


    def train(self):
        # Use GPU-based sampling and vectorized trajectory simulation
        init_states = sample_in_region_torch(self.num_paths_sampled, self.lb_tensor, self.ub_tensor, self.device)
        traj, values, _ = self.simulate_trajectories(init_states, max_steps=3000)
        # Here, 'values' is assumed to be the integrated norm computed during simulation.
        values = values.to(dtype=torch.float32, device=self.device)

        # 1) Enforce W(0) = 0
        zeros_tensor = torch.zeros((1, self.state_dim), dtype=torch.float32, device=self.device)
        W_zeros, _ = self.lyapunov_value(zeros_tensor)
        Lz = 5 * torch.square(W_zeros)

        # print(f'W(0) {self.lyapunov_value(zeros_tensor)}')
        # print(f'Lz loss: {Lz}')

        # 2) Enforce W(x) = tanh(alpha * V(x))
        Wx_Lr, _ = self.lyapunov_value(init_states)
        target = torch.tanh(self.alpha_zubov * values)
        Lr = F.mse_loss(Wx_Lr, target)

        # 3) Physics-Informed Loss (PDE residual)
        init_states_in = sample_in_region_torch(self.batch_size, self.lb_tensor, self.ub_tensor, self.device)
        Wx_in, grad_Wx_in = self.lyapunov_value(init_states_in, requires_grad=True)

        current_actions = self._get_current_policy_actions(init_states_in)
        current_fxu = self.dynamics_fn(init_states_in, current_actions)

        phix = torch.linalg.vector_norm(init_states_in, ord=2, dim=1)

        resid = torch.sum(grad_Wx_in * current_fxu.detach(), dim=1) + \
            self.alpha_zubov * (1 + Wx_in.squeeze()) * (1 - Wx_in.squeeze()) * phix
        Lp = torch.mean(torch.square(resid))

        # 4) Encourage control actions that decrease the Lyapunov function
        # grad_norm = torch.linalg.vector_norm(grad_Wx, ord=2, dim=1, keepdim=True)
        # unit_grad = grad_Wx / (grad_norm + 1e-8)
        # Lc = torch.mean(torch.sum(unit_grad.detach() * fxu, dim=1))
        Lc = torch.mean(torch.sum(grad_Wx_in.detach() * current_fxu, dim=1))

        # 5) Enforce that on the boundary of R2, W(x) ≈ 1
        init_states_out = sample_out_of_region_torch(self.batch_size, self.lb_tensor, self.ub_tensor, scale=2, device=self.device)
        Wx_out, _ = self.lyapunov_value(init_states_out)
        Lb = 2 * torch.mean(torch.abs(Wx_out - 1.0))

        actor_loss = Lc
        critic_loss = Lz + Lr + Lp + Lb

        if False:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.actor_scheduler.step()
            self.critic_scheduler.step()

        if True:
            total_loss = 0.5 * (actor_loss + critic_loss)
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step()

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
                u = self._get_current_policy_actions(x)
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
            Z, _ = self.lyapunov_value(grid_tensor)
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
        plt.savefig(f"./plots/lyAC_{self.timesteps}.png")
        plt.close()

    # dReal-specific functions
    def _sanity_network(self):
        from util.dreal import dreal_var
        x_np = np.random.randn(self.state_dim)
        x_d  = dreal_var(self.state_dim)
        y_pt = self.actor_model(torch.as_tensor(x_np, dtype=torch.float32,
                                                device=self.device).unsqueeze(0))[0].item()
        y_dr = self.actor_model.forward_dreal(x_d)[0]
        fsat = d.And(*[x_d[i] == x_np[i] for i in range(self.state_dim)],
                     y_dr >= y_pt - 1e-3,
                     y_dr <= y_pt + 1e-3)
        assert d.CheckSatisfiability(fsat, 1e-4).is_unsat()

    def _sanity_dynamics(self):
        from util.dreal import dreal_var
        from util.dynamics import pendulum_dynamics_dreal
        x_np = np.random.randn(self.state_dim)
        u_np = np.random.randn(1)
        x_d  = dreal_var(self.state_dim)
        u_d  = dreal_var(1, "u")
        f_pt = self.dynamics_fn(torch.as_tensor(x_np).unsqueeze(0),
                                torch.as_tensor(u_np).unsqueeze(0))[0].cpu().numpy()
        f_dr = pendulum_dynamics_dreal(x_d, u_d)
        cons = [x_d[i] == x_np[i] for i in range(self.state_dim)] + \
               [u_d[0] == u_np[0]] + \
               [f_dr[i] >= f_pt[i]-1e-3 for i in range(self.state_dim)] + \
               [f_dr[i] <= f_pt[i]+1e-3 for i in range(self.state_dim)]
        assert d.CheckSatisfiability(d.And(*cons), 1e-4).is_unsat()

    def check_lyapunov(self, level=0.9, scale=2.0, eps=0.5, delta=1e-4):
        """
        r1, r2 are dReal Results.  Both must be UNSAT for a certificate.
        """
        from util.dreal import dreal_var, in_box, on_boundary
        from util.dynamics import pendulum_dynamics_dreal

        x   = dreal_var(self.state_dim)
        u   = self.actor_model.forward_dreal(x)
        fx  = pendulum_dynamics_dreal(x, u)
        Wx  = self.critic_model.forward_dreal(x)[0]
        W0  = float(
            self.lyapunov_value(torch.zeros((1, self.state_dim), device=self.device))[0]
            )

        lie = sum(fx[i] * Wx.Differentiate(x[i]) for i in range(self.state_dim))
        xnorm = d.sqrt(sum(x[i]*x[i] for i in range(self.state_dim)))

        bad_inside = d.And(
            xnorm >= eps,
            in_box(x, self.lb, self.ub, scale),
            Wx <= level,
            d.Or(lie >= 0, Wx <= W0)
        )
        leak = d.And(on_boundary(x, self.lb, self.ub, scale), Wx <= level)

        r1 = d.CheckSatisfiability(bad_inside, delta)
        r2 = d.CheckSatisfiability(leak, delta)
        return r1, r2
