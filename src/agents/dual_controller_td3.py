import math
import numpy as np
import scipy.linalg

from agents.abstract_agent import AbstractAgent
from agents.td3_agent import TD3Agent
from agents.lqr_agent import LQRAgent


class DualControllerTD3(AbstractAgent):
    def __init__(self, config):
        super().__init__(config)

        self.td3_agent = TD3Agent(config)
        self.lqr_agent = LQRAgent(config)

        self.beta = config.get("beta", 0.8)
        self.s = np.arctanh(self.beta)

        self.state_dim = self.lqr_agent.P.shape[0]

        self.x_star = np.zeros(self.state_dim, dtype=np.float64)
        self.dynamics_func = config.get("dynamics_func")

        # Offline estimation of DoA
        Nv = config.get("doa_samples_nv", 5000)
        delta_c = config.get("doa_step_delta_c", 0.1)
        delta_threshold = config.get("doa_violation_threshold", 0.05)

        self.L_cholesky = scipy.linalg.cholesky(self.lqr_agent.P, lower=True)

        self.c_star = self._estimate_LQR_domain_of_attraction(delta_c, Nv, delta_threshold)
        print(f"Domain of Attraction estimation complete. Estimated c* = {self.c_star:.4f}")

    def _lyapunov_value(self, state):
        """Computes V(x) = (x - x*)^T P (x - x*) [cite: 137]."""
        delta_x = state.flatten() - self.x_star
        # Ensure P is accessible and correctly shaped
        P = self.lqr_agent.P
        v_x = delta_x @ P @ delta_x
        # Check for NaN before returning
        if math.isnan(v_x):
             print(f"Warning: NaN detected in Lyapunov calculation for state {state}. Returning infinity.")
             return float('inf')
        if v_x < 0:
             print(f"Warning: Negative Lyapunov value detected for state {state}.")
        return max(0, v_x)
    
    def _sample_from_ellipsoid(self, c, Nv):
        if c <= 0:
            return np.array([self.x_star])
        
        Z = np.random.randn(self.Nv, self.state_dim)

        # Transoform using Cholesky decomposition P = L L^T
        # We need inv(L^T) * Z, equivalent to solving L^T * Y = Z for Y
        # Y = scipy.linalg.solve_triangular(self.L_cholesky.T, Z.T, lower=False).T

        try:
            Y = scipy.linalg.solve_triangular(self.L_cholesky.T, Z.T, lower=False).T
        except np.linalg.LinAlgError:
            print("Error solving traingular system during sampling. Check Cholesky decomposition.")
            return np.array([self.x_star] * self.Nv)
        
        norms_z_squares = np.sum(Z**2, axis=1)
        norms_z_squares[norms_z_squares == 0] = 1e-6

        Y_normalized = Y / np.sqrt(norms_z_squares)[:, np.newaxis]

        Y_scaled = Y_normalized * np.sqrt(c)

        X_samples = Y_scaled + self.x_star
        return X_samples
            
    
    def _estimate_LQR_domain_of_attraction(self, delta_c, Nv, delta_threshold, max_c=1000.0):
        c_estimated = 0.0
        c = delta_c

        max_fails = int(delta_threshold * Nv)

        print(f"Estimating DoA: Nv={Nv}, delta_c={delta_c}, threshold={delta_threshold * 100:.1f}% ({max_fails} fails)")

        while c < max_c:
            n_fails = 0
            sampled_states = self._sample_from_ellipsoid(c, Nv)

            for i in range(Nv):
                xi = sampled_states[i]
                Vi = self._lyapunov_value(xi)

                u = self.lqr_agent.policy(xi)
                print(u)

                try:
                    x_prime = self.dynamics_func(xi, u)
                except Exception as e:
                    print(f"Error during dynamics simulation at c={c:.4f}, state={xi}, u={u}: {e}")
                    n_fails + 1
                    continue

                Vi_prime = self._lyapunov_value(x_prime)
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

    def compute_blending_factor(self, state):
        V = self._lyapunov_value(state)
        if self.c_star == 0:
            return 1.0
        v_norm = V / self.c_star
        return math.tanh(self.s * v_norm)

    def add_transition(self, transition):
        """
        Add a transition to the replay buffer (delegated to the TD3 agent).
        """
        self.td3_agent.add_transition(transition)

    def update(self) -> tuple:
        """
        Update the agent (we update the TD3 network only, as the LQR component is fixed).
        """
        loss = self.td3_agent.update()
        return loss

    def policy(self, state):
        """
        Return the blended action: LQR + h(x)*(TD3 - LQR).
        Assumes state is a NumPy array.
        """
        action_td3 = self.td3_agent.policy(state)
        action_lqr = self.lqr_agent.policy(state)
        
        blending_factor = self.h1_func(state)
        
        dual_action = action_lqr + blending_factor * (action_td3 - action_lqr)
        return dual_action
    
    def save(self, file_path: str = './') -> None:
        """
        Save both TD3 and LQR models.
        """
        self.td3_agent.save(file_path)
        self.lqr_agent.save(file_path)

    def load(self, file_path: str = './') -> None:
        """
        Load both TD3 and LQR models.
        """
        self.td3_agent.load(file_path)
        self.lqr_agent.load(file_path)
