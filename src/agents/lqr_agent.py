import os
import numpy as np
import pickle
from agents.abstract_agent import AbstractAgent

class LQRAgent(AbstractAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        self.g = config['g']    # acceleration due to gravity in m/s^2
        self.m = 1.0            # mass in kg
        self.l = 1.0            # length in m

        # For LQR design, we need a 2D state: [theta, theta_dot],
        # where theta = 0 is the upright position.
        # The environment observation is [cos(theta), sin(theta), theta_dot].
        # We then do theta = arctan2(sin(theta), cos(theta)).
        #
        # Linearize the pendulum dynamics about theta = 0.
        # The nonlinear dynamics are:
        #   theta_dot = theta_dot,
        #   theta_ddot = - (g/l)*sin(theta) + (1/(m*l^2))*u.
        # For small theta, sin(theta) ≈ theta, so:
        #   theta_ddot ≈ - (g/l)*theta + (1/(m*l^2))*u.
        #
        # Therefore, the system matrices are:
        A = np.array([[0, 1],
                      [(3 * self.g) / (2 * self.l), 0]])
        B = np.array([[0],
                      [3 / (self.m * self.l**2)]])
        
        self.A = A
        self.B = B
        
        # Cost matrices: Q = I (penalize state deviation equally) and R = 1 (penalize control effort)
        # I used the same value as Wang and Fazlyab (2024) to replicate their experiment
        Q = config.get("Q", np.eye(2))
        R = config.get("R", np.eye(1))

        # Solve the continuous-time Algebraic Riccati Equation
        # Equation: A^T P + P A - P B R^{-1} B^T P + Q = 0
        P = self.solve_continuous_are(A, B, Q, R)
        self.P = P
        
        # Compute the optimal gain
        # Gain is computed as: K = R^{-1} B^T P
        self.K = np.linalg.inv(R) @ (B.T @ P)  # shape (1,2)
        # print("LQR P matrix:", P)
        # print("LQR gain K:", self.K)

    def add_transition(self, transition: tuple) -> None:
        # LQR is computed offline; no transitions needed
        pass

    def update(self) -> None:
        # LQR does not learn, no updates needed
        return None

    def policy(self, state) -> np.array:
        cos_theta, sin_theta, theta_dot = state
        theta = np.arctan2(sin_theta, cos_theta)

        x = np.array([theta, theta_dot])
        u = -float(self.K @ x)

        u = np.clip(u, -2.0, 2.0)
        return np.array([u], dtype=np.float32)

    def save(self, file_path: str = './saved_models/') -> None:
        os.makedirs(file_path, exist_ok=True)
        with open(file_path + "lqr_agent.pkl", "wb") as f:
            pickle.dump({
                'K': self.K,
                'A': self.A,
                'B': self.B,
                'P': self.P,
                'g': self.g,
                'l': self.l
            }, f)

    def load(self, file_path: str = './saved_models/') -> None:
        with open(file_path + "lqr_agent.pkl", "rb") as f:
            data = pickle.load(f)
            self.K = data['K']
            self.A = data['A']
            self.B = data['B']
            self.P = data['P']
            self.g = data['g']
            self.l = data['l']

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

    def ellipse_area(self, c=1.0):
        """
        Returns area of the ellipse { x : x^T P x <= c } in 2D.
        The formula is pi * c / sqrt(det(P)), assuming P is positive-definite.
        """
        detP = np.linalg.det(self.P)
        if detP <= 0:
            return 0.0
        return np.pi * c / np.sqrt(detP)
