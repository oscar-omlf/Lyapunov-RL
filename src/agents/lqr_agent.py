import numpy as np
from scipy.linalg import solve_continuous_are
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
                      [(self.g/self.l), 0]])  # For g=10, l=1, A = [[0,1], [-10,0]]
        B = np.array([[0],
                      [1/(self.m * self.l**2)]])  # For m=1, l=1, B = [[0], [1]]
        
        self.A = A
        self.B = B
        
        # Cost matrices: Q = I (penalize state deviation equally) and R = 1 (penalize control effort)
        # I used the same value as Wang, J., and Fazlyab, M. (2024) to replicate their experiment.
        Q = np.eye(2)
        R = np.eye(1)

        # Solve the continuous-time Algebraic Riccati Equation
        # Equation: A^T P + P A - P B R^{-1} B^T P + Q = 0.
        P = solve_continuous_are(A, B, Q, R)
        self.P = P
        
        # Compute the optimal gain
        # Gain is computed as: K = R^{-1} B^T P.
        self.K = np.linalg.inv(R) @ (B.T @ P)  # shape (1,2)
        print("LQR P matrix:", P)
        print("LQR gain K:", self.K)

    def add_transition(self, transition: tuple) -> None:
        # LQR is computed offline; no transitions needed.
        pass

    def update(self) -> None:
        # LQR does not learn, no updates needed.
        return None

    def policy(self, state) -> np.array:
        cos_theta, sin_theta, theta_dot = state
        theta = np.arctan2(sin_theta, cos_theta)
        
        # Convert theta so that 0 corresponds to the upright position.
        if theta < 0:
            theta += 2 * np.pi
        theta_error = theta
        
        x = np.array([theta_error, theta_dot])
        u = -float(self.K @ x)

        # Clamp to the action space [-2, 2]        
        u = np.clip(u, -2.0, 2.0)
        return np.array([u], dtype=np.float32)

    def save(self, file_path: str = '../saved_models/') -> None:
        with open(file_path + "lqr_agent.pkl", "wb") as f:
            pickle.dump({
                'K': self.K,
                'A': self.A,
                'B': self.B,
                'P': self.P,
                'g': self.g,
                'l': self.l
            }, f)

    def load(self, file_path: str = '../saved_models/') -> None:
        with open(file_path + "lqr_agent.pkl", "rb") as f:
            data = pickle.load(f)
            self.K = data['K']
            self.A = data['A']
            self.B = data['B']
            self.P = data['P']
            self.g = data['g']
            self.l = data['l']
