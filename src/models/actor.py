import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Distribution

from models.twoheadedmlp import TwoHeadedMLP

class NStepActor(TwoHeadedMLP):
    """
    An actor network for n-step actor-critic.
    This network outputs the parameters of a multivariate Gaussian distribution.
    One output head predicts the mean and the other outputs the log-diagonal (for the Cholesky factor)
    of the covariance matrix, ensuring numerical stability.
    """
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize the NStepActor network.

        :param state_dim: Dimension of the state input.
        :param action_dim: Dimension of the action space.
        """
        super(NStepActor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(128, action_dim)  # Head for mean
        # Head for the log of the diagonal elements of the Cholesky decomposition
        self.log_diag_chol_head = nn.Linear(128, action_dim)

        # Optional: use double precision if desired (mimicking your one-step actor)
        self.double()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that computes the mean and the log-diagonal Cholesky factors.

        :param x: Input tensor of shape (batch_size, state_dim).
        :return: Tuple (mean, log_diag_chol) each of shape (batch_size, action_dim).
        """
        features = self.layers(x)
        mean = self.mean_head(features)
        log_diag_chol = self.log_diag_chol_head(features)
        return mean, log_diag_chol

    def predict(self, x: torch.Tensor) -> Distribution:
        """
        Given an input state, output a MultivariateNormal distribution representing the policy.

        :param x: Input tensor of shape (batch_size, state_dim).
        :return: A MultivariateNormal distribution parameterized by the network outputs.
        """
        mean, log_diag_chol = self(x)
        # Exponentiate the log diagonal to get the diagonal of the Cholesky factor,
        # then reconstruct the covariance matrix.
        cov_matrix = torch.diag_embed(torch.exp(log_diag_chol))
        return MultivariateNormal(mean, cov_matrix)
