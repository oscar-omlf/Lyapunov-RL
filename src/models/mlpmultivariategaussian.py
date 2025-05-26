import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Distribution
from typing import Tuple


from models.twoheadedmlp import TwoHeadedMLP


class MLPMultivariateGaussian(TwoHeadedMLP):

    def __init__(self, input_size: int, output_size: int):
        super(MLPMultivariateGaussian, self).__init__()
        self.layers = nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(128, output_size)  # Linear layer for mean
        # https://en.wikipedia.org/wiki/Cholesky_decomposition
        self.log_diag_chol_head = torch.nn.Linear(128,
                                                  output_size) # Linear layer for log diagonal of Cholesky decomposition
                                                               # The log is for numerical stability.

        # self.double()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.layers(x)
        mean = self.mean_head(features)
        log_diag_chol = self.log_diag_chol_head(features)
        return mean, log_diag_chol

    def predict(self, x: torch.Tensor) -> Distribution:
        mean, log_diag_chol = self(x)

        # Construct the covariance matrix using Cholesky decomposition
        cov_matrix = torch.diag_embed(torch.exp(log_diag_chol))

        return MultivariateNormal(mean, cov_matrix)
