import numpy as np
import scipy.linalg
import torch

from agents.abstract_agent import AbstractAgent
from agents.lqr_agent import LQRAgent
from agents.lyapunov_agent import LyapunovAgent
from util.sampling import sample_from_ellipsoid
from util.blending_function import BlendingFunction

class LAS_LyACAgent(AbstractAgent):
    def __init__(self, config):
        super().__init__(config)

        lqr_config = config["LQR"]
        lac_config = config["LAC"]

        self.lqr_agent = LQRAgent(lqr_config)

        self.beta = config.get("beta", 0.9)
        self.s = np.arctanh(self.beta)

        self.state_dim = self.state_space.shape[0]

        self.x_star = np.zeros(self.state_dim, dtype=np.float64)
        self.dynamics_func = config.get("dynamics_func")
        
        # Offline estimation of DoA
        Nv = config.get("doa_samples_nv", 5000)
        delta_c = config.get("doa_step_delta_c", 0.1)
        delta_threshold = config.get("doa_violation_threshold", 0.05)

        self.L_cholesky = scipy.linalg.cholesky(self.lqr_agent.P, lower=True)

        self.c_star = self._estimate_LQR_domain_of_attraction(delta_c, Nv, delta_threshold)
        print(f"Domain of Attraction estimation complete. Estimated c* = {self.c_star:.4f}")

        self.blending_function = BlendingFunction(self.lqr_agent, beta_h=self.beta, c_star=self.c_star, device=self.device)

        lac_config["dual_controller_components"] = {
            "LQR": self.lqr_agent,
            "BLENDING_FUNCTION": self.blending_function
        }

        self.lac_agent = LyapunovAgent(lac_config)

    def _estimate_LQR_domain_of_attraction(self, delta_c, Nv, delta_threshold, max_c=1000.0):
        c_estimated = 0.0
        c = delta_c

        max_fails = int(delta_threshold * Nv)

        print(f"Estimating DoA: Nv={Nv}, delta_c={delta_c}, threshold={delta_threshold * 100:.1f}% ({max_fails} fails)")

        while c < max_c:
            n_fails = 0
            sampled_np = sample_from_ellipsoid(c, Nv, self.x_star, self.L_cholesky, self.state_dim)
            sampled_t = torch.from_numpy(sampled_np).to(dtype=torch.float32, device=self.device)

            for i in range(Nv):
                xi = sampled_t[i]
                Vi = self.lqr_agent.lyapunov_value(xi)

                u = self.lqr_agent.policy(xi)

                try:
                    x_prime = self.dynamics_func(xi, u)
                except Exception as e:
                    print(f"Error during dynamics simulation at c={c:.4f}, state={xi}, u={u}: {e}")
                    n_fails + 1
                    continue

                Vi_prime = self.lqr_agent.lyapunov_value(x_prime)
                if Vi_prime - Vi > 1e-6:
                    n_fails += 1
                    continue

                if n_fails > max_fails:
                    print(f"Warning: Exceeded maximum number of fails ({max_fails}) for c={c:.4f}")
                    break

            c_estimated = c
            c += delta_c

        print(f"Estimated DoA: c* = {c_estimated:.4f}")
        return c_estimated

    def add_transition(self, transition):
        pass

    def update(self) -> tuple:
        loss = self.lac_agent.update()
        return loss
    
    def policy(self, state):
        """
        Return the blended action: pi_theta(x) = pi_loc(x) + pi_glo(x).
        Where pi_glo(x) is defined as pi_glo(x) = h1(x)*(pi_td3(x) - pi_loc(x)).
        Assumes state is a NumPy array.
        """
        action_lac = self.lac_agent.policy(state)
        action_lqr = self.lqr_agent.policy(state)
        
        blending_factor = self.blending_function.get_h1(state)
        
        dual_action = action_lqr + blending_factor * (action_lac - action_lqr)
        return dual_action
    
    def load(self, path: str) -> None:
        pass

    def save(self, path: str) -> None:
        pass