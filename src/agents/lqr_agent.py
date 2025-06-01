import math
import numpy as np
import torch
from scipy.linalg import expm

from agents.abstract_agent import AbstractAgent
from util.riccati_solver import RiccatiSolver


class LQRAgent(AbstractAgent):
    def __init__(self, config: dict):
        super().__init__(config)

        self.riccati_solver = RiccatiSolver()

        self.environment = config.get("environment", "InvertedPendulum")
        self.discrete = config.get("discrete", False)

        self.max_action = config.get("max_action", 1.0)
        self.dt = config.get("dt", 0.03)

        self.state_dim = self.state_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        # For LQR design, we need a 2D state: [theta, theta_dot],
        
        # Linearize original pendulum dynamics
        # around theta = 0: sin(theta) ~ theta
        if self.environment == "InvertedPendulum":
            self.g = config['g']
            self.m = config['m']
            self.l = config['l']

            A_c = np.array([
                [0.0, 1.0],
                [self.g / self.l, 0.0]
            ], dtype=np.float64)

            B_c = np.array([
                [0.0],
                [1.0 / (self.m * self.l**2)]
            ], dtype=np.float64)

        elif self.environment == "VanDerPol":
            self.mu = config.get("mu", 1.0)

            A_c = np.array([
                [0.0, 1.0],
                [-1.0, self.mu]
            ], dtype=np.float64)

            B_c = np.array([
                [0.0],
                [1.0]
            ], dtype=np.float64)
        else:
            raise ValueError(f"Unknown environment: {self.environment}")
        
        if self.discrete:
            A, B = self.discretize_dynamics(A_c, B_c, dt=self.dt)
        else:
            A, B = A_c, B_c
        
        # Cost matrices: Q = I (penalize state deviation equally) and R = I (penalize control effort)
        # I used the same value as Wang and Fazlyab (2024) to replicate their experiment
        Q = config.get("Q", np.eye(self.state_dim, dtype=np.float64))
        R = config.get("R", np.eye(self.action_dim, dtype=np.float64))

        if self.discrete:
            P = self.riccati_solver.solve_discrete_are(A, B, Q, R)
            K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        else:
            P = self.riccati_solver.solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ (B.T @ P)

        # Store the Numpy version of the matrices
        self.A_np = A
        self.B_np = B
        self.P_np = P
        self.K_np = K

        # Store the torch version of the matrices
        self.A = torch.from_numpy(A).to(dtype=torch.float32, device=self.device)
        self.B = torch.from_numpy(B).to(dtype=torch.float32, device=self.device)
        self.P = torch.from_numpy(P).to(dtype=torch.float32, device=self.device)
        self.K = torch.from_numpy(K).to(dtype=torch.float32, device=self.device)

        self.x_star_np = np.zeros(self.state_dim, dtype=np.float64)
        self.x_star = torch.zeros(self.state_dim, dtype=torch.float32, device=self.device)

        # print("LQR P matrix:", P)
        # print("LQR gain K:", self.K)

    def discretize_dynamics(self, A_c: np.ndarray, B_c: np.ndarray, dt: float):
        n = A_c.shape[0]
        m = B_c.shape[1]

        # Form the augmented matrix for ZOH computation
        # M = [[A_c, B_c],
        #      [0_mxn, 0_mxm]]
        # Md = expm(M * dt)
        # A_d = Md[:n, :n]
        # B_d = Md[:n, n:]

        # Ensure B_c is 2D
        if B_c.ndim == 1:
            B_c = B_c.reshape(-1, 1)
        
        augmented_matrix = np.zeros((n + m, n + m))
        augmented_matrix[:n, :n] = A_c
        augmented_matrix[:n, n:] = B_c
        
        # Compute matrix exponential
        phi = expm(augmented_matrix * dt)
        
        A_d = phi[:n, :n]
        B_d = phi[:n, n:]
        
        return A_d, B_d
    
    def policy(self, state) -> torch.Tensor:
        """Compute control u = -K * x, where x = [theta, theta_dot]."""
        x = state.to(dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        u = -(x @ self.K.T)

        u = torch.clamp(u, -self.max_action, self.max_action)
        return u

    def policy_np(self, state) -> np.array:
        """
        Compute control u = -K * x, where x = [theta, theta_dot].
        """
        x = np.array(state, dtype=np.float64).reshape(-1, self.state_dim) # (B, n)
        u = -x.dot(self.K_np.T)                                      # (B,1)
        
        u = np.clip(u, -self.max_action, self.max_action).astype(np.float32)
        return u

    def lyapunov_value(self, state) -> torch.Tensor:
        """Computes V(x) = (x - x*)^T P (x - x*) in torch."""
        x = state.float()

        x_star = torch.from_numpy(self.x_star.astype(np.float32)).to(device=self.device)
        delta = x - x_star

        V = (delta @ self.P) * delta
        V = V.sum(dim=-1)

        V = torch.where(torch.isnan(V),
                        torch.full_like(V, float('inf')),
                        V)
        V = torch.clamp(V, min=0.0)

        return V

    def lyapunov_value_np(self, state):
        """Computes V(x) = (x - x*)^T P (x - x*)."""
        x = np.array(state, dtype=np.float64).flatten()
        delta_x = x - self.x_star
        v_x = float(delta_x @ self.P_np @ delta_x)

        if math.isnan(v_x):
            print(f"Warning: NaN detected in Lyapunov calculation for state {state}. Returning infinity.")
            v_x = float('inf')
        if v_x < 0:
             print(f"Warning: Negative Lyapunov value detected for state {state}.")
             v_x = 0.0
        return v_x
    
    def save(self, file_path: str = './saved_models/') -> None:
        pass

    def load(self, file_path: str = './saved_models/') -> None:
        pass

    def add_transition(self, transition: tuple) -> None:
        # LQR is computed offline; no transitions needed
        pass

    def update(self) -> None:
        # LQR does not learn, no updates needed
        pass
    