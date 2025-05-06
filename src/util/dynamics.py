import torch


def pendulum_dynamics(xs, us):
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


def original_pendulum_dynamics(xs, us):
    theta = xs[:, 0]
    thetad = xs[:, 1]

    g = 9.81
    m = 0.15
    l = 0.5

    theta_ddot = g / l * torch.sin(theta) - (3.0 / (m * l**2)) * us.squeeze()

    dtheta = thetad
    dtheta_ddot = theta_ddot

    dxdt = torch.stack([dtheta, dtheta_ddot], dim=1)
    return dxdt