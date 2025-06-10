import numpy as np
import matplotlib.pyplot as plt
import torch

from agents.abstract_agent import AbstractAgent
from agents.lqr_agent import LQRAgent
from util.blending_function import BlendingFunction
from util.riccati_solver import RiccatiSolver

import dreal as d
from util.dreal import dreal_var, in_box, on_boundary, is_unsat


class DualPolicyAgent(AbstractAgent):
    def __init__(self, config):
        super().__init__(config)

        lqr_config = config.get("LQR")

        if lqr_config is None:
            raise ValueError("LQR configuration is missing.")
        
        self.lqr_agent = LQRAgent(lqr_config)

        self.beta = config.get("beta", 0.5)
        self.s = np.arctanh(self.beta)

        self.x_star = torch.zeros(self.state_dim, dtype=torch.float32, device=self.device)
        self.u_star = torch.zeros(self.action_dim, dtype=torch.float32, device=self.device)

        self.dynamics_fn = config.get("dynamics_fn")
        self.dynamics_fn_dreal = config.get("dynamics_fn_dreal")

        r1_bounds = config.get("r1_bounds")
        self.lb = r1_bounds[0]
        self.ub = r1_bounds[1]

        self.max_action = config.get("max_action")
        
        # Offline estimation of DoA
        self.c_star = self._estimate_domain_of_attraction(check_fn=self.lqr_check)
        print(f"Domain of Attraction estimation complete. Estimated c* = {self.c_star:.4f}")

        self.blending_function = BlendingFunction(self.lqr_agent, beta_h=self.beta, c_star=self.c_star, device=self.device)
        self.riccati_solver = RiccatiSolver()
        
        self.actor_model = None
        self.critic_model = None

        self.timesteps = 0

    def add_transition(self, transition):
        pass

    def update(self) -> tuple:
        loss = self._train()
        return loss
    
    def policy(self, state):
        """
        Return the blended action: pi_theta(x) = pi_loc(x) + pi_glo(x).
        Where pi_glo(x) is defined as pi_glo(x) = h1(x)*(pi_td3(x) - pi_loc(x)).
        """
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)

            pi_loc = self._get_local_action(state_t)
            pi_glo = self._get_global_action(state_t)

            h1_val = self.blending_function.get_h1(state_t)

            if h1_val.ndim == 1:
                h1_val = h1_val.unsqueeze(-1)

            blended_action = pi_loc + h1_val * (pi_glo - pi_loc)
            final_action = torch.clamp(blended_action, -self.max_action, self.max_action)

            return final_action.cpu().numpy().flatten()

    def _get_blended_action(self, state_torch: torch.Tensor, mu_theta_action_torch: torch.Tensor) -> torch.Tensor:
        pi_loc_batch_torch = self._get_local_action(state_torch)

        h1_val = self.blending_function.get_h1(state_torch)
        if h1_val.ndim == 1:
            h1_val = h1_val.unsqueeze(-1)

        blended_actions = pi_loc_batch_torch + h1_val * (mu_theta_action_torch - pi_loc_batch_torch)
        blended_actions = torch.clamp(blended_actions, -self.max_action, self.max_action)
        return blended_actions

    def _get_local_action(self, state_torch: torch.Tensor) -> torch.Tensor:
        k_error = -(state_torch @ self.lqr_agent.K.T)
        pi_loc_actions = self.u_star.unsqueeze(0) + k_error
        pi_loc_actions = torch.clamp(pi_loc_actions, -self.max_action, self.max_action)
        return pi_loc_actions
    
    def _get_global_action(self, state_torch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method.")

    def load(self, file_path) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

    def save(self, file_path) -> None:
        raise NotImplementedError("Subclasses must implement this method.")


    def lqr_check_discrete(self, level, scale=2., eps=0.5, delta=1e-4):
        n = self.lqr_agent.P_np.shape[0]
        x = dreal_var(n)

        P = self.lqr_agent.P_np
        V = sum(x[i] * sum(P[i, j] * x[j] for j in range(n)) for i in range(n))

        K = self.lqr_agent.K_np
        u = [-sum(K[i, j] * x[j] for j in range(n)) for i in range(K.shape[0])]

        x_next = self.dynamics_fn_dreal(x, u)
        V_next = sum(x_next[i] * sum(P[i, j] * x_next[j] for j in range(n)) for i in range(n))
        
        lyapunov_violation = (V_next - V >= 0)
        x_norm_sq = sum(xi**2 for xi in x)

        r1 = d.CheckSatisfiability(
            d.And(
                V <= level,
                x_norm_sq >= eps**2,
                in_box(x, self.lb, self.ub, scale),
                lyapunov_violation
            ),
            delta)

        r2 = d.CheckSatisfiability(
            d.And(
                on_boundary(x, self.lb, self.ub, scale),
                V <= level
            ),
            delta)

        return r1, r2

    def lqr_check_continuous(self, level, scale=2., eps=1e-5, delta=1e-3):
        n = self.lqr_agent.P_np.shape[0]
        x = dreal_var(n)

        P = self.lqr_agent.P_np
        V = sum(x[i] * sum(P[i, j] * x[j] for j in range(n)) for i in range(n))

        K = self.lqr_agent.K_np
        u_lqr = [-sum(K[i, j] * x[j] for j in range(n)) for i in range(K.shape[0])]
        
        u_clamped = [
            d.if_then_else(u_lqr[i] > self.max_action, self.max_action, 
                d.if_then_else(u_lqr[i] < -self.max_action, -self.max_action, u_lqr[i]))
            for i in range(self.action_dim)
        ]

        fxu = self.dynamics_fn_dreal(x, u_clamped)

        lie_derivative = sum(fxu[i] * V.Differentiate(x[i]) for i in range(n))

        x_norm_sq = sum(xi**2 for xi in x)


        violation_condition = d.And(
            V <= level,
            x_norm_sq >= eps**2,
            in_box(x, self.lb, self.ub, scale),
            lie_derivative >= 0
        )
        r1 = d.CheckSatisfiability(violation_condition, delta)

        boundary_condition = d.And(
            on_boundary(x, self.lb, self.ub, scale),
            V <= level
        )
        r2 = d.CheckSatisfiability(boundary_condition, delta)

        return r1, r2
    
    def lqr_check(self, level, scale=2., eps=0.5, delta=1e-4, alpha=0.2):
        if self.lqr_agent.discrete_discounted:
            return self.lqr_check_discrete(level, scale, eps, delta)
        else:
            return self.lqr_check_continuous(level, scale, eps, delta)

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
