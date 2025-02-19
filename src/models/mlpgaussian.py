import torch
import torch.nn as nn
from torch.distributions import Normal

from models.twoheadedmlp import TwoHeadedMLP


class MLPGaussian(TwoHeadedMLP):
    """
    A multilayer perceptron with two output heads representing a Gaussian distribution.
    This neural network outputs a Gaussian distribution, where one head is for the mean and the other
    is for the variance.
    NOTE: This neural network should be equivalent o the Multivariate Gaussian
    variant when the output dimension is 1. I have included this class for educational purposes.
    """

    def __init__(self, input_size: int):
        """
        Initialize the MLP with two output heads.

        :param input_size: Number of input features.
        """
        super(MLPGaussian, self).__init__()
        self.layers = nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
        )
        self.mean_head = torch.nn.Linear(128, 1)  # Linear layer for mean
        self.log_std_head = torch.nn.Linear(128, 1)  # log std instead of variance for numerical stability.

        self.double()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the multi-layer perceptron.

        :param x: Input tensor of shape (batch_size, input_size).
        :return: Tuple containing mean tensor of shape (batch_size, output_size)
        and log standard deviation tensor of shape (batch_size, output_size).
        """
        features = self.layers(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        return mean, log_std

    def predict(self, x: torch.Tensor) -> torch.distributions.Distribution:
        """
        Predict a Normal distribution given an input tensor.

        :param x: Input tensor of shape (batch_size, input_size).
        :return: Normal distribution.

        The forward pass gives a mean and log standard deviation.
        This function constructs a Normal distribution using them.
        """
        mean, log_std = self(x)

        return Normal(mean, torch.exp(log_std))
