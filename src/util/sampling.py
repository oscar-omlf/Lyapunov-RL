import numpy as np
import scipy.linalg
import torch
from typing import Tuple


from models.mlpmultivariategaussian import TwoHeadedMLP
from torch.distributions import Distribution


def sample_two_headed_gaussian_model(model: TwoHeadedMLP, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample an action from a Gaussian policy modeled by the provided model and compute the log probability of the action.

    :param model: The TwoHeadedMLP model representing the policy.
    :param state: The input state tensor.
    :return: A tuple containing the sampled action tensor and the log probability of the action.
    """
    action_distribution: Distribution = model.predict(state)
    action: torch.Tensor = action_distribution.sample()
    ln_prob: torch.Tensor = action_distribution.log_prob(action)
    return action, ln_prob


def log_prob_policy(model: TwoHeadedMLP, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """
    Compute the log probability of an action given a state and a model representing the policy.

    :param model: The TwoHeadedMLP model representing the policy.
    :param state: The input state tensor.
    :param action: The action tensor.
    :return: The log probability of the action given the state and policy model.
    """
    distribution: Distribution = model.predict(state)

    return distribution.log_prob(action)


def sample_in_region(num_samples: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.random.uniform(lb, ub, size=(num_samples, lb.shape[0]))


def sample_out_of_region(num_samples: int, lb: np.ndarray, ub: np.ndarray, scale: float = 2.0) -> np.ndarray:
    x = np.random.uniform(-1, 1, size=(num_samples, lb.shape[0]))
    # For each sample, compute the maximum ratio: |x_i| / (ub_i * scale)
    ratios = np.max(np.abs(x) / (ub * scale), axis=1, keepdims=True)
    # Scale the sample so that at least one coordinate is at the boundary
    x = x / ratios
    # Add small noise in the same sign direction to push samples outside
    noise = np.random.uniform(0, 0.5, size=x.shape)
    x = x + np.sign(x) * noise
    return x


def sample_in_region_torch(num_samples: int, lb: torch.Tensor, ub: torch.Tensor, device: str) -> torch.Tensor:
    return torch.rand(num_samples, lb.shape[0], device=device) * (ub - lb) + lb


def sample_out_of_region_torch(num_samples: int, lb: torch.Tensor, ub: torch.Tensor, scale: float, device: str) -> torch.Tensor:
    x = torch.rand(num_samples, lb.shape[0], device=device) * 2 - 1  
    ratios = torch.max(torch.abs(x) / (ub * scale), dim=1, keepdim=True).values
    x = x / ratios
    noise = torch.rand_like(x) * 0.5
    x = x + torch.sign(x) * noise
    return x


def sample_in_lqr_ellipsoid_torch(
    num_samples: int, 
    c_star: float,
    L_inv: torch.Tensor,
    device: str
) -> torch.Tensor:
    state_dim = L_inv.shape[0]
    L_inv = L_inv.to(device)

    z = torch.randn(num_samples, state_dim, device=device)
    
    z_norm = z / torch.linalg.vector_norm(z, ord=2, dim=1, keepdim=True)
    
    u = torch.rand(num_samples, 1, device=device)
    r = u.pow(1.0 / state_dim)
    
    points_in_unit_sphere = z_norm * r
    points_in_unit_ellipsoid = points_in_unit_sphere @ L_inv
    
    return np.sqrt(c_star) * points_in_unit_ellipsoid


def sample_on_circle_boundary_torch(
    num_samples: int, 
    state_dim: int,
    r2_radius: float, 
    device: str
) -> torch.Tensor:
    z = torch.randn(num_samples, state_dim, device=device)
    z_norm = z / torch.linalg.vector_norm(z, ord=2, dim=1, keepdim=True)
    
    return z_norm * r2_radius
