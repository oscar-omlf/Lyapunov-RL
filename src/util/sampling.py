import numpy as np
import scipy.linalg
import torch
from models.mlpmultivariategaussian import TwoHeadedMLP
from torch.distributions import Distribution


def sample_two_headed_gaussian_model(model: TwoHeadedMLP, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


def sample_in_region_gpu(num_samples: int, lb: torch.Tensor, ub: torch.Tensor, device: str) -> torch.Tensor:
    return torch.rand(num_samples, lb.shape[0], device=device) * (ub - lb) + lb


def sample_out_of_region_gpu(num_samples: int, lb: torch.Tensor, ub: torch.Tensor, scale: float, device: str) -> torch.Tensor:
    x = torch.rand(num_samples, lb.shape[0], device=device) * 2 - 1  
    ratios = torch.max(torch.abs(x) / (ub * scale), dim=1, keepdim=True).values
    x = x / ratios
    noise = torch.rand_like(x) * 0.5
    x = x + torch.sign(x) * noise
    return x


def sample_from_ellipsoid(c, Nv, x_star, L_cholesky, state_dim):
    if c <= 0:
        return np.array([x_star])
    
    Z = np.random.randn(Nv, state_dim)

    # Transoform using Cholesky decomposition P = L L^T
    # We need inv(L^T) * Z, equivalent to solving L^T * Y = Z for Y
    # Y = scipy.linalg.solve_triangular(self.L_cholesky.T, Z.T, lower=False).T

    try:
        Y = scipy.linalg.solve_triangular(L_cholesky.T, Z.T, lower=False).T
    except np.linalg.LinAlgError:
        print("Error solving traingular system during sampling. Check Cholesky decomposition.")
        return np.array([x_star] * Nv)
    
    norms_z_squares = np.sum(Z**2, axis=1)
    norms_z_squares[norms_z_squares == 0] = 1e-6

    Y_normalized = Y / np.sqrt(norms_z_squares)[:, np.newaxis]

    Y_scaled = Y_normalized * np.sqrt(c)

    X_samples = Y_scaled + x_star
    return X_samples
