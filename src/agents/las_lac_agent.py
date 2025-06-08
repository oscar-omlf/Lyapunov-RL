import os
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR

from agents.abstract_agent import AbstractAgent
from agents.lqr_agent import LQRAgent
from util.blending_function import BlendingFunction
from models.lyapunov_actor import LyapunovActor
from models.lyapunov_critic import LyapunovCritic
from util.sampling import sample_in_region_torch, sample_out_of_region_torch
from util.rk4_step import rk4_step

import dreal as d
from util.dreal import dreal_var, in_box, on_boundary, is_unsat


class LAS_LACAgent(AbstractAgent):
    def __init__(self, config):
        super().__init__(config)

        lqr_config = config["LQR"]
        self.lqr_agent = LQRAgent(lqr_config)

        self.ub = config.get("r1_bounds")[1]
        self.lb = config.get("r1_bounds")[0]
        self.lb_tensor = torch.tensor(
            self.lb, 
            dtype=torch.float32, 
            device=self.device
        )
        self.ub_tensor = torch.tensor(
            self.ub, 
            dtype=torch.float32, 
            device=self.device
        )

        self.beta = config.get("beta", 0.5)
        self.s = np.arctanh(self.beta)

        self.state_dim = self.state_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        self.x_star = np.zeros(self.state_dim, dtype=np.float64)
        self.dynamics_func = config.get("dynamics_fn")
        self.dynamics_func_dreal = config.get("dynamics_fn_dreal")
        
        # Offline estimation of DoA
        self.c_star = self._estimate_LQR_domain_of_attraction(check_fn=self.lqr_check)
        print(f"Domain of Attraction estimation complete. Estimated c* = {self.c_star:.4f}")

        self.blending_function = BlendingFunction(self.lqr_agent, beta_h=self.beta, c_star=self.c_star, device=self.device)

        ### Lyapunov Agent
        self.alpha_zubov = config.get("alpha")
        self.lr = config.get("lr")
        self.dynamics_fn = config.get("dynamics_fn")
        self.dynamics_fn_dreal = config.get("dynamics_fn_dreal")
        self.batch_size = config.get("batch_size")
        self.num_paths_sampled = config.get("num_paths_sampled")
        self.dt = config.get("dt")
        self.norm_threshold = config.get("norm_threshold")
        self.integ_threshold = config.get("integ_threshold")
        self.r1_bounds = config.get("r1_bounds")

        actor_hidden_sizes = config.get("actor_hidden_sizes")
        critic_hidden_sizes = config.get("critic_hidden_sizes")

        self.max_action = config.get("max_action")
        
        self.actor_model = LyapunovActor(
            input_size=self.state_dim, 
            hidden_sizes=actor_hidden_sizes, 
            action_dim=self.action_dim, 
            max_action=self.max_action
        ).to(device=self.device)

        self.critic_model = LyapunovCritic(
            input_size=self.state_dim, 
            hidden_sizes=critic_hidden_sizes
        ).to(device=self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor_model.parameters()) + list(self.critic_model.parameters()), 
            lr=self.lr
        )
        self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.8)

        self.timesteps = 0

        print('Dual Controller Initialized (Lyapunov)!')

    def add_transition(self, transition):
        pass

    def update(self) -> tuple:
        loss = self._train()
        return loss
    
    def policy(self, state):
        """
        Return the blended action: pi_theta(x) = pi_loc(x) + pi_glo(x).
        Where pi_glo(x) is defined as pi_glo(x) = h1(x)*(pi_td3(x) - pi_loc(x)).
        Assumes state is a NumPy array.
        """
        combined_policy = self._get_combined_policy(state)
        return combined_policy

    def load(self, file_path: str = './saved_models/') -> None:
        self.lac_agent.load(file_path)

    def save(self, file_path: str = './saved_models/') -> None:
        self.lac_agent.save(file_path)



    def lqr_check(self, level, scale=2., eps=0.5, delta=1e-4, alpha=0.2):
        x = dreal_var(2)
        V = sum(x[i]*sum(self.lqr_agent.P_np[i,j]*x[j] for j in range(2)) for i in range(2))
        W = d.tanh(alpha * V)
        u = - (self.lqr_agent.K_np @ np.array(x))[0]
        fx = self.dynamics_func_dreal(x, np.array([u]))
        lie = sum(fx[i] * W.Differentiate(x[i]) for i in range(2))
        xnorm = d.sqrt(x[0]*x[0]+x[1]*x[1])

        W0 = 0.0

        r1 = d.CheckSatisfiability(
                d.And(xnorm>=eps, in_box(x, self.lb, self.ub, scale),
                        W<=level, d.Or(lie>=0, W<=W0)),
                delta)
        r2 = d.CheckSatisfiability(
                d.And(on_boundary(x, self.lb, self.ub, scale), W<=level), delta)
        return r1, r2

    def _estimate_domain_of_attraction(self, check_fn, c_max=0.95, tol=1e-3, it_max=40):
        hi_fail = c_max
        while True:
            r1, r2 = check_fn(hi_fail, eps=0.5)
            if is_unsat(r1) and is_unsat(r2):
                break
            hi_fail *= 0.5
            if hi_fail < 1e-6:
                print("No certificate even at c = 0.")
                return 0.0
            print(hi_fail)

        lo_pass = hi_fail
        hi_fail = hi_fail * 2.0

        for _ in range(it_max):
            if hi_fail - lo_pass < tol:
                break
            mid = 0.5 * (lo_pass + hi_fail)
            r1, r2 = check_fn(mid)
            ok = is_unsat(r1) and is_unsat(r2)
            print(f"c={mid:.5f}  ok={ok}")
            if ok:
                lo_pass = mid
            else:
                hi_fail = mid
        return lo_pass
    
    def lyapunov_value(self, state: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        """
        Computes the Lyapunov/ Zubov value W(x) of the combined model.
        Returns W(x) and optionally its gradient w.r.t. state.
        """
        if requires_grad and not state.requires_grad:
            state.requires_grad_(True)

        Wx_learned = self.critic_model(state)

        V_loc = self.blending_function.get_lyapunov_value(state)
        W_loc = torch.tanh(self.alpha_zubov * V_loc)

        h2 = self.blending_function.get_h2(state)
        Wx_final = W_loc + h2 * (Wx_learned - W_loc)

        grad_Wx = None
        if requires_grad:
            if Wx_final.requires_grad:
                grad_Wx = torch.autograd.grad(Wx_final.sum(), state, create_graph=True, retain_graph=True)[0] 
            else:
                print('Wx_final does not require grad. Setting grad_Wx to zero.')
                grad_Wx = torch.zeros_like(state)

        return Wx_final, grad_Wx
    
    def _get_combined_policy(self, state):
        """
        Helper to get actions from the combined policy.
        """
        actions_learned = self.actor_model(state)

        actions_loc = self.lqr_agent.policy(state)

        h1_blend = self.blending_function.get_h1(state)

        if h1_blend.ndim == 1:
            h1_blend = h1_blend.unsqueeze(-1)

        combined_actions = actions_loc + h1_blend * (actions_learned - actions_loc)

        return combined_actions
    

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
    

    def _sanity_network(self):
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
        assert d.CheckSatisfiability(d.And(*cons), 1e-4).is_unsat()

    def check_lyapunov(self, level=0.9, scale=2., eps=0.5):
        x = dreal_var(self.state_dim)
        
        # 1. Composite Lyapunov Function (W_composite)
        # Compute W_loc = tanh(alpha * V_local)
        delta_x = [x[i] - self.lqr_agent.x_star[i] for i in range(self.state_dim)]
        V_local = sum(delta_x[i] * sum(self.lqr_agent.P_np[i][j] * delta_x[j] for j in range(2)) for i in range(2))
        W_loc = d.tanh(self.alpha_zubov * V_local)
        
        # Compute W_learned (critic output)
        W_learned = self.critic_model.forward_dreal(x)[0]
        
        # Compute h2 blending
        v_x = V_local / self.blending_function.c_star  # Normalized V_local
        h2 = d.tanh(self.blending_function.s_factor * (v_x)**(3/2))
        
        # Composite W
        W_composite = W_loc + h2 * (W_learned - W_loc)
        
        # 2. Composite Policy Action
        action_lqr = -sum(self.lqr_agent.K_np[0][i] * delta_x[i] for i in range(2))
        action_learned = self.actor_model.forward_dreal(x)[0]
        
        # Compute h1 blending
        h1 = d.tanh(self.blending_function.s_factor * v_x)
        action_composite = action_lqr + h1 * (action_learned - action_lqr)
        
        # 3. Dynamics with Composite Action
        fx = self.dynamics_fn_dreal(x, [action_composite])
        
        # 4. Lie Derivative of W_composite
        lie_derivative = sum(fx[i] * W_composite.Differentiate(x[i]) for i in range(2))
        
        # 5. Lyapunov Conditions
        x_norm = d.sqrt(sum(x[i]**2 for i in range(2)))
        W0 = 0.0  # W_composite(0) = 0

        condition = d.And(
            x_norm >= eps,
            in_box(x, self.lb, self.ub, scale),
            W_composite <= level,
            d.Or(lie_derivative >= 0, W_composite <= W0)
        )
        
        r1 = d.CheckSatisfiability(condition, 1e-4)
        
        r2 = d.CheckSatisfiability(
            d.And(
                on_boundary(x, self.lb, self.ub, scale),
                W_composite <= level
            ), 
            1e-4
        )
        
        print('---')
        print('r1', r1)
        print('---')
        print('r2', r2)
        print('---')

        return r1, r2
