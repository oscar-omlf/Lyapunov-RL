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

        self.dynamics_func = config.get("dynamics_fn")
        self.dynamics_func_dreal = config.get("dynamics_fn_dreal")

        r1_bounds = config.get("r1_bounds")
        self.lb = r1_bounds[0]
        self.ub = r1_bounds[1]
        
        # Offline estimation of DoA
        self.c_star = self._estimate_domain_of_attraction(check_fn=self.lqr_check)
        print(f"Domain of Attraction estimation complete. Estimated c* = {self.c_star:.4f}")

        self.blending_function = BlendingFunction(self.lqr_agent, beta_h=self.beta, c_star=self.c_star, device=self.device)
        self.riccati_solver = RiccatiSolver()

        self.dynamics_fn = config.get("dynamics_fn")
        self.dynamics_fn_dreal = config.get("dynamics_fn_dreal")

        self.max_action = config.get("max_action")
        
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
        Assumes state is a NumPy array.
        """
        actions_learned = self.actor_model(state)
        actions_loc = self.u_star + self.lqr_agent.policy(state)

        h1_blend = self.blending_function.get_h1(state)

        if h1_blend.ndim == 1:
            h1_blend = h1_blend.unsqueeze(-1)

        combined_actions = actions_loc + h1_blend * (actions_learned - actions_loc)

        return combined_actions

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

    def lqr_check(self, level, scale=2., eps=1e-5, delta=1e-3):
        """
        Checks if the Lyapunov level set V(x) <= level defines a valid Region of Attraction.

        This function uses the dReal SMT solver to search for a counterexample
        to the discrete-time Lyapunov stability conditions.

        Args:
            level (float): The candidate value c* for the level set V(x) <= c*.
            scale (float): Scaling factor for state bounds.
            eps (float): A small tolerance to exclude the origin from the check.
            delta (float): Precision for the dReal solver.

        Returns:
            tuple: A tuple (r1, r2) containing the results from dReal.
                - r1: Result for the Lyapunov decrease condition. UNSAT means no violation was found.
                - r2: Result for the state bounds check. UNSAT means the level set is within bounds.
        """
        # Get the dimension of the state space
        n = self.lqr_agent.P_np.shape[0]
        
        # 1. Define symbolic state variables
        x = dreal_var(n)

        # 2. Define the quadratic Lyapunov function V(x) = x'Px
        # This assumes the setpoint x* is at the origin, so x_bar = x.
        P = self.lqr_agent.P_np
        V = sum(x[i] * sum(P[i, j] * x[j] for j in range(n)) for i in range(n))

        # 3. Define the local stabilizing policy u = -Kx
        K = self.lqr_agent.K_np
        u = [-sum(K[i, j] * x[j] for j in range(n)) for i in range(K.shape[0])]

        # 4. Get the symbolic expression for the next state from the dynamics
        x_next = self.dynamics_func_dreal(x, u)

        # 5. Define the Lyapunov function at the next state, V(x_next)
        V_next = sum(x_next[i] * sum(P[i, j] * x_next[j] for j in range(n)) for i in range(n))
        
        # Condition for the violation of Lyapunov stability decrease [cite: 141]
        lyapunov_violation = (V_next - V >= 0)

        # Use squared norm to avoid sqrt, which is better for SMT solvers
        x_norm_sq = sum(xi**2 for xi in x)

        # --- SMT QUERIES ---

        # Query 1 (r1): Search for a violation of the Lyapunov decrease condition.
        # We are looking for a state x such that:
        # 1. It is inside the candidate level set (V(x) <= level).
        # 2. It is NOT the origin (x_norm_sq >= eps**2)[cite: 141].
        # 3. It is within the overall valid state space (in_box).
        # 4. The Lyapunov condition is violated.
        # If this is UNSAT, the level set is stable.
        r1 = d.CheckSatisfiability(
            d.And(
                V <= level,
                x_norm_sq >= eps**2,
                in_box(x, self.lb, self.ub, scale),
                lyapunov_violation
            ),
            delta)

        # Query 2 (r2): Check if the level set stays within the defined state bounds.
        # We are looking for a state x such that:
        # 1. It is on the boundary of the valid state space (on_boundary).
        # 2. It is also inside our candidate level set (V(x) <= level).
        # If this is UNSAT, the level set is safely contained within the state bounds.
        r2 = d.CheckSatisfiability(
            d.And(
                on_boundary(x, self.lb, self.ub, scale),
                V <= level
            ),
            delta)

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
