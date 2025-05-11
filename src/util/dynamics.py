import numpy as np
import torch


def gym_pendulum_dynamics(xs, us):
    """
    Differentiable dynamics function for the pendulum.
    Assumes xs has shape (batch, 2): [theta, theta_dot],
    and us has shape (batch, 1).
    Returns dx/dt with the same shape as xs.
    """
    theta = xs[:, 0]
    theta_dot = xs[:, 1]
    
    # Constants for Gymnasium's Pendulum-v1
    g = 10.0  # gravitational acceleration
    m = 1.0   # mass
    l = 1.0   # pendulum length
    
    # Compute angular acceleration:
    # d(theta_dot)/dt = 3 * g / (2 * l) * sin(theta) + 3 / (m * l^2) * u
    theta_ddot = (3 * g / (2 * l)) * torch.sin(theta) + (3.0 / (m * l**2)) * us.squeeze()
    
    # Derivatives:
    dtheta = theta_dot
    dtheta_dot = theta_ddot
    
    dxdt = torch.stack([dtheta, dtheta_dot], dim=1)
    return dxdt


def pendulum_dynamics_torch(
    state: torch.Tensor,
    action: torch.Tensor,
    g: float = 9.81,
    m: float = 0.15,
    l: float = 0.5
) -> torch.Tensor:
    """
    Returns the time derivative dx/dt of the pendulum state x = [theta, theta_dot].
    Supports state of shape (..., 2) and scalar or broadcastable action.
    """
    # state[..., 0] = theta, state[..., 1] = theta_dot
    theta = state[..., 0]
    theta_dot = state[..., 1]

    # compute angular acceleration
    theta_ddot = (g / l) * torch.sin(theta) - (3.0 / (m * l**2)) * action

    # pack the derivatives
    dtheta = theta_dot
    dtheta_ddot = theta_ddot

    # stack back into (..., 2)
    dxdt = torch.stack([dtheta, dtheta_ddot], dim=-1)
    return dxdt


def pendulum_dynamics_np(state: np.ndarray, action: float, g: float = 9.81, m: float = 0.15, l: float = 0.5) -> np.ndarray:
    """
    Returns the time derivative dx/dt of the pendulum state x = [theta, theta_dot].
    """
    theta, theta_dot = state

    theta_ddot = (g / l) * np.sin(theta) - (3.0 / (m * l**2)) * action

    dtheta = theta_dot
    dtheta_ddot = theta_ddot

    dxdt = np.array([dtheta, dtheta_ddot], dtype=np.float64)
    return dxdt


def compute_pendulum_reward(state: np.ndarray, action: float) -> float:
    """
    Returns the cost for the given state and action using:
    reward = theta**2 + 0.1 * theta_dot**2 + 0.001 * (action**2)
    """
    theta, theta_dot = state
    cost = theta**2 + 0.1 * theta_dot**2 + 0.001 * (action**2)
    return -cost