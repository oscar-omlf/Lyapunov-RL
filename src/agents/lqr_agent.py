import os
import math
import numpy as np
import pickle
import torch

from agents.abstract_agent import AbstractAgent


class LQRAgent(AbstractAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        self.g = config['g']    # acceleration due to gravity in m/s^2
        self.m = config['m']    # mass in kg
        self.l = config['l']    # length in m

        self.max_action = config.get("max_action", 1.0)

        state_dim = self.state_space.shape[0]
        action_dim = self.action_space.shape[0]

        self.x_star = config.get("x_star", np.zeros(state_dim, dtype=np.float64))


        # For LQR design, we need a 2D state: [theta, theta_dot],
        # where theta = 0 is the upright position.
        #
        
        # Linearize original pendulum dynamics:
        #   theta_dot = theta_dot
        #   theta_ddot = (g/l)*sin(theta) - (3/(m*l^2))*u
        # around theta = 0: sin(theta) ~ theta
        A = np.array([
            [0.0, 1.0],
            [self.g / self.l, 0.0]
        ], dtype=np.float64)

        B = np.array([
            [0.0],
            [1.0 / (self.m * self.l**2)]
        ], dtype=np.float64) 
        
        # Cost matrices: Q = I (penalize state deviation equally) and R = I (penalize control effort)
        # I used the same value as Wang and Fazlyab (2024) to replicate their experiment
        Q = config.get("Q", np.eye(state_dim, dtype=np.float64))
        R = config.get("R", np.eye(action_dim, dtype=np.float64))

        # Solve the continuous-time Algebraic Riccati Equation
        # Equation: A^T P + P A - P B R^{-1} B^T P + Q = 0
        P = self.solve_continuous_are(A, B, Q, R)

        # Compute the optimal gain
        # Gain is computed as: K = R^{-1} B^T P
        K = np.linalg.inv(R) @ (B.T @ P)  # shape (1, state_dim)

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

        # print("LQR P matrix:", P)
        # print("LQR gain K:", self.K)
    
    def policy(self, state) -> torch.Tensor:
        """Compute control u = -K * x, where x = [theta, theta_dot]."""
        x = state.float()
        u = -self.K @ x

        u = torch.clamp(u, -self.max_action, self.max_action)
        return u

    def policy_np(self, state) -> np.array:
        """
        Compute control u = -K * x, where x = [theta, theta_dot].
        """
        x = np.array(state, dtype=np.float64).reshape(-1, 2)         # (B,2)
        u = -x.dot(self.K_np.T)                                      # (B,1)
        
        u = np.clip(u, -self.max_action, self.max_action).astype(np.float32)
        return u

    def solve_continuous_are(self, A, B, Q, R):
        """
        Solve the continuous-time algebraic Riccati equation (CARE):
        A^T P + P A - P B R^{-1} B^T P + Q = 0
        using an eigenvalue method.
        
        Parameters:
            A (ndarray): System matrix (n x n).
            B (ndarray): Input matrix (n x m).
            Q (ndarray): State cost matrix (n x n), should be symmetric positive semi-definite.
            R (ndarray): Control cost matrix (m x m), should be symmetric positive definite.
        
        Returns:
            P (ndarray): The unique symmetric positive semi-definite solution to the CARE.
        """
        n = A.shape[0]
        R_inv = np.linalg.inv(R)
        
        # Form the Hamiltonian matrix
        H = np.block([[A, -B @ R_inv @ B.T],
                    [-Q, -A.T]])
        
        # Compute eigenvalues and eigenvectors of the Hamiltonian
        eigenvalues, eigenvectors = np.linalg.eig(H)
        
        # Identify the indices of eigenvalues with negative real parts
        stable_indices = np.where(np.real(eigenvalues) < 0)[0]
        
        if len(stable_indices) != n:
            raise ValueError("The number of stable eigenvalues does not equal the system dimension n.")
        
        # Extract the eigenvectors corresponding to the stable eigenvalues
        stable_eigenvectors = eigenvectors[:, stable_indices]
        
        # Partition the eigenvectors into two n x n blocks: X (top half) and Y (bottom half)
        X = stable_eigenvectors[:n, :]
        Y = stable_eigenvectors[n:, :]
        
        # Ensure X is invertible; if not, numerical issues have occurred
        if np.linalg.matrix_rank(X) < n:
            raise np.linalg.LinAlgError("The matrix X is not invertible.")
        
        # Compute the solution P = Y * inv(X)
        P = np.real(Y @ np.linalg.inv(X))
        
        return P
    
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
        return None