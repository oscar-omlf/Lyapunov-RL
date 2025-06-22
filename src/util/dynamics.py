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
    if state.ndim == 1:
        x1, x2 = state[0], state[1]
        u = action[0]
        x1_dot = x2
        x2_dot = -x1 + mu * (1 - x1 * x1) * x2 + u
        return torch.stack([x1_dot, x2_dot])
    
    elif state.ndim == 2:
        x1 = state[:, 0]
        x2 = state[:, 1]
        u = action[:, 0]

        x1_dot = x2
        x2_dot = -x1 + mu * (1 - x1 * x1) * x2 + u

        dxdt = torch.stack([x1_dot, x2_dot], dim=1)
        return dxdt


def vanderpol_dynamics_np(state: np.ndarray, action: np.ndarray, mu: float = 1.0) -> np.ndarray:
    if state.ndim == 1:
        state = state.reshape(1, -1)
    if action.ndim == 0:
        action = np.array([action]).reshape(1, -1)
    elif action.ndim == 1 and action.shape[0] == state.shape[0]:
        action = action.reshape(-1,1)

    x1 = state[:, 0]
    x2 = state[:, 1]
    u = action[:, 0]

    x1_dot = x2
    x2_dot = -x1 + mu * (1 - x1 * x1) * x2 + u

    dxdt = np.stack([x1_dot, x2_dot], axis=1)
    return dxdt

def vanderpol_dynamics_dreal(state_vars, action_vars, mu=1.0):
    x1, x2 = state_vars[0], state_vars[1]
    u = action_vars[0]

    x1_dot = x2
    x2_dot = -x1 + mu * (1 - x1 * x1) * x2 + u

    dxdt = np.array([x1_dot, x2_dot])
    return dxdt

def compute_vanderpol_reward(state: np.ndarray, action: float) -> float:
    x1, x2 = state

    state_cost = x1**2 + x2**2
    action_cost = 0.1 * action**2

    success_bonus = 0
    if np.sqrt(state_cost) < 0.1:
        success_bonus = 1

    cost = success_bonus - state_cost - action_cost
    return cost


def bicycle_dynamics_torch(
        state: torch.Tensor, 
        action: torch.Tensor,
        v: float = 5.0,
        l: float = 1.0,
    ) -> torch.Tensor:
    if state.ndim == 1:
        x1, x2 = state[0], state[1]
        u = action[0]

        y1 = v * torch.sin(x2)
        y2 = v * torch.tan(u) / l - torch.cos(x2) / (1 - x1)

        return torch.stack([y1, y2])
    
    elif state.ndim == 2:
        x1 = state[:, 0]
        x2 = state[:, 1]
        u = action[:, 0]

        y1 = v * torch.sin(x2)
        y2 = v * torch.tan(u) / l - torch.cos(x2) / (1 - x1)

        dxdt = torch.stack([y1, y2], dim=1)
        return dxdt
    

def bicycle_dynamics_np(
        state: np.ndarray, 
        action: np.ndarray,
        v: float = 5.0,
        l: float = 1.0,
    ) -> np.ndarray:
    if state.ndim == 1:
        state = state.reshape(1, -1)
    if action.ndim == 0:
        action = np.array([action]).reshape(1, -1)
    elif action.ndim == 1 and action.shape[0] == state.shape[0]:
        action = action.reshape(-1, 1)

    x1 = state[:, 0]
    x2 = state[:, 1]
    u = action[:, 0]

    y1 = v * np.sin(x2)
    y2 = v * np.tan(u) / l - np.cos(x2) / (1 - x1)

    dydt = np.stack([y1, y2], axis=1)
    return dydt


def bicycle_dynamics_dreal(
        state_vars,
        action_vars,
        v: float = 5.0,
        l: float = 1.0,
    ):
    x1, x2 = state_vars[0], state_vars[1]
    u = action_vars[0]

    y1 = v * d.sin(x2)
    y2 = v * d.tan(u) / l - d.cos(x2) / (1 - x1)

    dydt = np.array([y1, y2])
    return dydt
