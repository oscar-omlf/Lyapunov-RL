# Utility functions below, assuming the above model is used for approximating a policy.
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
