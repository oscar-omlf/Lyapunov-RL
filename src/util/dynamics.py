import numpy as np
import torch
import dreal as d


def pendulum_dynamics_torch(
    state: torch.Tensor,
    action: torch.Tensor,
    g: float = 9.81,
    m: float = 0.15,
    l: float = 0.5
) -> torch.Tensor:
    theta = state[..., 0]
    theta_dot = state[..., 1]

    action = action.squeeze(-1)

    # theta_ddot = (g / l) * torch.sin(theta) - (3.0 / (m * l**2)) * action 
    # theta_ddot = (g / l) * torch.sin(theta) + (1.0 / (m * l**2)) * action
    
    # with friction b = 0.1
    b = 0.1
    theta_ddot = (g / l) * torch.sin(theta) - (b / (m * l * l)) * theta_dot + (1.0 / (m * l * l)) * action

    dtheta = theta_dot
    dtheta_ddot = theta_ddot

    dxdt = torch.stack([dtheta, dtheta_ddot], dim=-1)
    return dxdt


def pendulum_dynamics_np(
        state: np.ndarray, 
        action: np.ndarray, 
        g: float = 9.81, 
        m: float = 0.15, 
        l: float = 0.5
) -> np.ndarray:
    if state.ndim == 1: # Single sample
        state = state.reshape(1, -1)
    if action.ndim == 0: # Single scalar action
        action = np.array([action]).reshape(1, -1)
    elif action.ndim == 1 and action.shape[0] == state.shape[0]: # Batch of scalar actions
        action = action.reshape(-1,1)

    theta = state[:, 0]
    theta_dot = state[:, 1]
    u = action[:, 0]

    # d(theta_dot)/dt = g / l * sin(theta) + 1 / (m * l^2) * u
    # theta_ddot = (g / l) * np.sin(theta) + (1.0 / (m * l**2)) * u
    b = 0.1
    theta_ddot = (g / l) * np.sin(theta) - (b / (m * l * l)) * theta_dot + (1.0 / (m * l * l)) * u

    dtheta = theta_dot
    dtheta_dot = theta_ddot

    dxdt = np.stack([dtheta, dtheta_dot], axis=1)
    return dxdt


def pendulum_dynamics_dreal(
    state,
    action,
    g: float = 9.81,
    m: float = 0.15,
    l: float = 0.5
):
    theta, omega = state
    action = action[0]

    theta_dot  = omega
    # theta_ddot  = g / l * d.sin(theta) - (3.0 / (m * l**2)) * action
    # theta_ddot = (g / l) * d.sin(theta) + (1.0 / (m * l**2)) * action
    b = 0.1
    theta_ddot = (g / l) * d.sin(theta) - (b / (m * l * l)) * omega + (1.0 / (m * l * l)) * action

    dtheta = theta_dot
    dtheta_dot = theta_ddot

    dxdt = np.array([dtheta, dtheta_dot])
    return dxdt


def compute_pendulum_reward(state: np.ndarray, action: float) -> float:
    theta, theta_dot = state
    reward = theta**2 + 0.1 * theta_dot**2 + 0.001 * (action**2)
    return -reward
