import torch

def pendulum_dynamics(xs, us):
    """
    Differentiable dynamics function for Gymnasium's Pendulum-v1.
    Assumes xs has shape (batch, 3): [cos(theta), sin(theta), theta_dot],
    and us has shape (batch, 1).
    Returns dx/dt with the same shape as xs.
    """
    # Recover theta from [cos(theta), sin(theta)]
    theta = torch.atan2(xs[:, 1], xs[:, 0])
    theta_dot = xs[:, 2]
    
    # For Gymnasium's Pendulum-v1 (https://gymnasium.farama.org/environments/classic_control/pendulum/):
    g = 10.0  # gravitational acceleration
    m = 1.0   # mass
    l = 1.0   # pendulum length
    
    # Gymnasium's dynamics update for angular acceleration:
    # dtheta_dot/dt = 3 * g / (2 * l) * sin(theta) + 3.0 / (m * l**2) * u
    theta_ddot = (3 * g / (2 * l)) * torch.sin(theta) + (3.0 / (m * l**2)) * us.squeeze()
    
    # Compute derivatives for the trigonometric state components:
    # d(cos(theta))/dt = -sin(theta) * theta_dot
    dcos = -torch.sin(theta) * theta_dot
    # d(sin(theta))/dt = cos(theta) * theta_dot
    dsin = torch.cos(theta) * theta_dot
    # d(theta_dot)/dt = theta_ddot
    dtheta_dot = theta_ddot
    
    # For a continuous-time derivative (dx/dt)
    dxdt = torch.stack([dcos, dsin, dtheta_dot], dim=1)
    return dxdt