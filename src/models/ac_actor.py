import torch
import torch.nn as nn
from torch.distributions import Normal

from models.twoheadedmlp import TwoHeadedMLP

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
class ACActor(TwoHeadedMLP):
    def __init__(self, input_size, hidden_sizes=(64,64), action_dim=1):
        super(ACActor, self).__init__()
        layers = []
        prev_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.layers = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

    def forward(self, x):
        features = self.layers(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def predict(self, x):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        return Normal(mean, std)
        