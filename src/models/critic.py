import torch
import torch.nn as nn

class NStepCritic(nn.Module):
    """
    A critic network for n-step actor-critic.
    This network takes a state as input and outputs a single scalar value
    representing the estimated value (or n-step return) for that state.
    """
    def __init__(self, state_dim: int):
        """
        Initialize the NStepCritic network.

        :param state_dim: Dimension of the state input.
        """
        super(NStepCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # self.double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that outputs the estimated state value.

        :param x: Input tensor of shape (batch_size, state_dim).
        :return: Tensor of shape (batch_size, 1) representing the state value.
        """
        return self.net(x)
