import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import dreal as d

from agents.abstract_agent import AbstractAgent
from trainers.abstract_trainer import Trainer
from util.sampling import sample_in_region_torch, sample_out_of_region_torch
from util.rk4_step import rk4_step
from util.dreal import dreal_var, in_box, on_boundary, is_unsat


class LAS_LyapunovAC_Trainer(Trainer):
    def __init__(
        self,
        agent: AbstractAgent,
        alpha_zubov: float,
        batch_size: int,
        num_paths_sampled: int,
        norm_threshold: float,
        integ_threshold: int,
        dt: float,
        dynamics_fn: callable,
        dynamics_fn_dreal: callable,
        state_dim: int,
        action_dim: int,
        r1_bounds: list,
        run_dir: str,
        device: str,
    ):
        super().__init__()
        self.agent = agent
        self.alpha_zubov = alpha_zubov
        self.batch_size = batch_size
        self.num_paths_sampled = num_paths_sampled
        self.norm_threshold = norm_threshold
        self.integ_threshold = integ_threshold
        self.dt = dt

        self.dynamics_fn = dynamics_fn
        self.dynamics_fn_dreal = dynamics_fn_dreal

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lb, self.ub = r1_bounds

        self.run_dir = run_dir

        self.device = device

        self.lb_tensor = torch.tensor(self.lb, dtype=torch.float32, device=self.device)
        self.ub_tensor = torch.tensor(self.ub, dtype=torch.float32, device=self.device)

        self.timesteps = 0

        print('Lyapunov Trainer Initialized (Dual-Policy Lyapunov AC)!')

    def train(self, counter_examples: list = None):
        init_states = sample_in_region_torch(self.num_paths_sampled, self.lb_tensor, self.ub_tensor, self.device)
        traj, values, _ = self.simulate_trajectories(init_states, max_steps=3000)
        values = values.to(dtype=torch.float32, device=self.device)

        # 1) Enforce W(0) = 0
        zero_tensor = torch.zeros((1, self.state_dim), device=self.device)
        W_zeros = self.agent.critic_model(zero_tensor)
        Lz = 5.0 * torch.square(W_zeros)

        # 2) Enforce W(x) = tanh(alpha * V(x))
        Wx_Lr = self.agent.get_composite_W_value(init_states)
        target = torch.tanh(self.alpha_zubov * values)
        Lr = F.mse_loss(Wx_Lr, target)

        # 3) Physics-Informed Loss (PDE residual)
        if counter_examples is not None and len(counter_examples) > 0:
            ce_tensor = torch.as_tensor(counter_examples, dtype=torch.float32, device=self.device)
            random_samples = sample_in_region_torch(self.batch_size - len(counter_examples), self.lb_tensor, self.ub_tensor, self.device)
            init_states_in = torch.cat([ce_tensor, random_samples], dim=0)
        else:
            init_states_in = sample_in_region_torch(self.batch_size, self.lb_tensor, self.ub_tensor, self.device)
        
        init_states_in.requires_grad_(True)
        Wx_in = self.agent.get_composite_W_value(init_states_in)

        grad_Wx_in, = torch.autograd.grad(
            outputs=Wx_in.sum(), 
            inputs=init_states_in, 
            create_graph=True
        )

        global_actions = self.agent._get_global_action(init_states_in)
        current_actions = self.agent._get_blended_action(init_states_in, global_actions)

        current_fxu = self.dynamics_fn(init_states_in, current_actions)

        phix = torch.norm(init_states_in, p=2, dim=1)

        resid = torch.sum(grad_Wx_in * current_fxu.detach(), dim=1) + \
            self.alpha_zubov * (1 + Wx_in.squeeze()) * (1 - Wx_in.squeeze()) * phix
        Lp = torch.mean(torch.square(resid))

        # 4) Encourage control actions that decrease the Lyapunov function
        Lc = 0.5 * torch.mean(torch.sum(grad_Wx_in.detach() * current_fxu, dim=1))

        # 5) Enforce that on the boundary of R2, W(x) \approx 1
        init_states_out = sample_out_of_region_torch(self.batch_size, self.lb_tensor, self.ub_tensor, scale=2, device=self.device)
        Wx_out = self.agent.get_composite_W_value(init_states_out)
        Lb = 5.0 * F.l1_loss(Wx_out, torch.ones_like(Wx_out).to(self.device))

        actor_loss = Lc
        critic_loss = Lz + Lr + Lp + Lb

        total_loss = 0.5 * actor_loss + critic_loss
        self.agent.optimizer.zero_grad()
        total_loss.backward()
        self.agent.optimizer.step()
        self.agent.scheduler.step()

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
                pi_glob = self.agent.actor_model(x)
                u = self.agent._get_blended_action(x, pi_glob)

                if u.dim() == 1:
                    u = u.unsqueeze(1)
                x_next = rk4_step(self.dynamics_fn, x, u, dt=self.dt)
                # Only update the active trajectories
                x = torch.where(active.unsqueeze(1), x_next, x)
                x_hist.append(x.clone())

        traj = torch.stack(x_hist, dim=1)
        final_norm = torch.linalg.vector_norm(x, ord=2, dim=1)
        converged = final_norm < self.norm_threshold

        return traj, integ_acc, converged
    
    def plot_level_set_and_trajectories(self):
        x_min, x_max = self.lb[0]*2, self.ub[0]*2
        y_min, y_max = self.lb[1]*2, self.ub[1]*2

        xx = np.linspace(x_min, x_max, 500)
        yy = np.linspace(y_min, y_max, 500)
        X, Y = np.meshgrid(xx, yy)
        grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
        grid_tensor = torch.as_tensor(grid_points, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            Z = self.agent.get_composite_W_value(grid_tensor)
            Z = Z.cpu().numpy()
        Z = Z.reshape(X.shape)
        
        plt.figure(figsize=(6,5))
        cp = plt.contourf(X, Y, Z, levels=100, cmap='viridis')
        plt.colorbar(cp)
        plt.title("Critic Level Set (Lyapunov Function)")
        plt.xlabel("x[0]")
        plt.ylabel("x[1]")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        init_states = sample_in_region_torch(5, self.lb_tensor, self.ub_tensor, self.device)
        trajs, _, _ = self.simulate_trajectories(init_states, max_steps=3000)
        trajs_np = trajs.cpu().numpy()
        
        for i in range(trajs_np.shape[0]):
            plt.plot(trajs_np[i, :, 0], trajs_np[i, :, 1], 'o-', markersize=1, linewidth=0.5, color='r')
            plt.plot(trajs_np[i, 0, 0], trajs_np[i, 0, 1], 'r+')
        
        plt.gca().set_aspect('equal')
        level_set_dir = os.path.join(self.run_dir, "level_sets")
        os.makedirs(level_set_dir, exist_ok=True)
        plot_path = os.path.join(level_set_dir, f"level_set_{self.timesteps}.png")
        plt.savefig(plot_path)
        plt.close()

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

    def check_lyapunov_with_ce(self, level=0.9, scale=2., eps=0.5):
        """
        Checks the Lyapunov conditions for the FULL COMPOSITE agent.
        This "translates" the original checker to the dual-policy framework.
        """
        print(f"Verifying COMPOSITE agent with c = {level:.4f} and eps = {eps:.2f}...")
        
        # Define dReal symbolic variable for the state
        x = dreal_var(self.state_dim)

        # === TRANSLATION STEP 1: Get the action u ===
        # Instead of just the global actor, we build the full composite policy.
        
        # 1a. Get local LQR policy: pi_loc = -Kx
        pi_loc_dreal = [-sum(self.agent.lqr_agent.K_np[i, j] * x[j] for j in range(self.state_dim)) for i in range(self.action_dim)]
        
        # 1b. Get global policy from the learned actor
        pi_glo_dreal = self.agent.actor_model.forward_dreal(x)
        
        # 1c. Get blending weight h1 symbolically
        V_loc_dreal = sum(x[i] * sum(self.agent.lqr_agent.P_np[i, j] * x[j] for j in range(self.state_dim)) for i in range(self.state_dim))
        v_norm_dreal = V_loc_dreal / self.agent.blending_function.c_star
        h1_dreal = d.tanh(self.agent.blending_function.s_factor.item() * v_norm_dreal)
        
        # 1d. The final blended action `u` is the composite policy
        u = [pi_loc_dreal[i] + h1_dreal * (pi_glo_dreal[i] - pi_loc_dreal[i]) for i in range(self.action_dim)]
        
        # === TRANSLATION STEP 2: Get the Lyapunov value Wx ===
        # Instead of just the global critic, we build the full composite W value.

        # 2a. Get global value from the learned critic
        W_glo_dreal = self.agent.critic_model.forward_dreal(x)[0]
        
        # 2b. Get local value W_loc = tanh(alpha * V_loc)
        W_loc_dreal = d.tanh(self.alpha_zubov * V_loc_dreal)

        # 2c. Get blending weight h2 symbolically
        h2_dreal = d.tanh(self.agent.blending_function.s_factor.item() * d.pow(v_norm_dreal, 1.5))
        
        # 2d. The final Lyapunov value `Wx` is the composite critic value
        Wx = W_loc_dreal + h2_dreal * (W_glo_dreal - W_loc_dreal)
        
        # === TRANSLATION STEP 3: Get W(0) ===
        # For the composite agent, W_comp(0) is always 0 by construction.
        W0 = 0.0

        # === STEP 4: Calculate Lie Derivative and check conditions ===
        # This logic remains the same, but now uses our composite `u` and `Wx`.
        fx = self.dynamics_fn_dreal(x, u)
        lie_derivative_W = sum(fx[i] * Wx.Differentiate(x[i]) for i in range(self.state_dim))
        x_norm = d.sqrt(sum(xi * xi for xi in x))

        # Condition 1: Find a violation of stability inside the level set
        violation_condition = d.And(
            x_norm >= eps,
            self.in_domain_dreal(x, scale), 
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

        boundary_condition = d.And(
            self.on_boundary_dreal(x, scale=scale),
            Wx <= level
        )
        r2 = d.CheckSatisfiability(boundary_condition, 0.001)
        if r2:
            print("FALSIFIED: Composite RoA level set touches the boundary of the verified region.")
            return (False, r2)

        print("Verification PASSED for the composite agent at this level.")
        return (True, None)
