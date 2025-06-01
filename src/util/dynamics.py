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
    theta = state[..., 0]
    theta_dot = state[..., 1]

    action = action.squeeze(-1)

    # theta_ddot = (g / l) * torch.sin(theta) - (3.0 / (m * l**2)) * action 
    theta_ddot = (g / l) * torch.sin(theta) + (1.0 / (m * l**2)) * action
    # b = 0.01
    # theta_ddot = (g / l) * torch.sin(theta) - (b / (m * l * l)) * theta_dot + (1.0 / (m * l**2)) * action

    dtheta = theta_dot
    dtheta_ddot = theta_ddot

    dxdt = torch.stack([dtheta, dtheta_ddot], dim=-1)
    return dxdt


def pendulum_dynamics_np(
        state: np.ndarray, 
        action: float, 
        g: float = 9.81, 
        m: float = 0.15, 
        l: float = 0.5
) -> np.ndarray:
    """
    Returns the time derivative dx/dt of the pendulum state x = [theta, theta_dot].
    Works with batched state and action.
    """
    theta = state[:, 0]
    theta_dot = state[:, 1]

    # d(theta_dot)/dt = 3 * g / (2 * l) * sin(theta) + 3 / (m * l^2) * u
    theta_ddot = (g / l) * np.sin(theta) - (3.0 / (m * l**2)) * action

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
    import dreal as d
    theta, omega = state
    action = action[0]

    theta_dot  = omega
    # theta_ddot  = g / l * d.sin(theta) - (3.0 / (m * l**2)) * action
    theta_ddot = (g / l) * d.sin(theta) + (1.0 / (m * l**2)) * action
    # b = 0.01
    # theta_ddot = (g / l) * d.sin(theta) - (b / (m * l * l)) * omega + (1.0 / (m * l * l)) * action

    dtheta = theta_dot
    dtheta_dot = theta_ddot

    dxdt = np.array([dtheta, dtheta_dot])
    return dxdt


def compute_pendulum_reward(state: np.ndarray, action: float) -> float:
    """
    Returns the cost for the given state and action using:
    reward = theta**2 + 0.1 * theta_dot**2 + 0.001 * (action**2)
    """
    theta, theta_dot = state
    cost = theta**2 + 0.1 * theta_dot**2 + 0.001 * (action**2)
    return -cost



def double_integrator_dynamics_torch(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    if state.ndim == 1:
        x1_dot = state[1]
        x2_dot = action[0]
        return torch.stack([x1_dot, x2_dot])
    elif state.ndim == 2:
        x1_dot = state[:, 1]
        x2_dot = action[:, 0]
        return torch.stack([x1_dot, x2_dot], dim=1)

def double_integrator_dynamics_np(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    
    if state.ndim == 1:
        x1_dot = state[1]
        x2_dot = action[0]
        return np.array([x1_dot, x2_dot])
    elif state.ndim == 2:
        x1_dot = state[:, 1]
        x2_dot = action[:, 0]
        return np.stack([x1_dot, x2_dot], axis=1)
    

def double_integrator_dynamics_dreal(state_vars, action_vars):
    x1, x2 = state_vars[0], state_vars[1]
    u = action_vars[0]

    x1_dot = x2
    x2_dot = u

    return [x1_dot, x2_dot]


def vanderpol_dynamics_torch(state: torch.Tensor, action: torch.Tensor, mu: float = 1.0) -> torch.Tensor:
    if state.ndim == 1:  # Single sample
        x1, x2 = state[0], state[1]
        u = action[0]
        
        x1_dot = x2
        x2_dot = x1 - mu * (1 - x1**2) * x2 + u
        return torch.stack([x1_dot, x2_dot])
    elif state.ndim == 2:  # Batch of samples
        x1 = state[:, 0]
        x2 = state[:, 1]
        u = action[:, 0] # Assuming action is (N,1) so u becomes (N,)
        
        x1_dot = x2
        x2_dot = x1 - mu * (1 - x1**2) * x2 + u
        return torch.stack([x1_dot, x2_dot], dim=1)


def vanderpol_dynamics_np(state: np.ndarray, action: np.ndarray, mu: float = 1.0) -> np.ndarray:
    if state.ndim == 1: # Single sample
        x1, x2 = state[0], state[1]
        u = action[0]
        
        x1_dot = x2
        x2_dot = x1 - mu * (1 - x1**2) * x2 + u
        return np.array([x1_dot, x2_dot])
    elif state.ndim == 2: # Batch of samples
        x1 = state[:, 0]
        x2 = state[:, 1]
        u = action[:, 0] # Assuming action is (N,1) so u becomes (N,)
        
        x1_dot = x2
        x2_dot = x1 - mu * (1 - x1**2) * x2 + u
        return np.stack([x1_dot, x2_dot], axis=1)
    

def vanderpol_dynamics_dreal(state_vars, action_vars, mu=1.0):
    x1, x2 = state_vars[0], state_vars[1]
    u = action_vars[0]

    # Dynamics
    x1_dot = x2
    x2_dot = x1 - mu * (1 - x1 * x1) * x2 + u

    return [x1_dot, x2_dot]