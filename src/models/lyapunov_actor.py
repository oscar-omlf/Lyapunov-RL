import torch
import torch.nn as nn
from torch.distributions import Normal

from models.mlp import MLP


class LyapunovActor(nn.Module):
    def __init__(self, input_size, hidden_sizes=(64, 64), action_dim=1):
        super(LyapunovActor, self).__init__()
        # Use the MLP as a feature extractor.
        self.feature_extractor = MLP(input_size, hidden_sizes, output_size=hidden_sizes[-1])
        # Separate heads for mean and log standard deviation.
        self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        return mean, log_std

    def predict(self, x):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        return Normal(mean, std)
