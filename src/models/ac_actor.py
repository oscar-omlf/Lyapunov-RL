import torch
import torch.nn as nn
from torch.distributions import Normal

from models.twoheadedmlp import TwoHeadedMLP
from models.mlp import MLP

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
class ACActor(TwoHeadedMLP):
    def __init__(self, input_size, hidden_sizes=(64, 64), action_dim=1):
        super(ACActor, self).__init__()
        self.feature_extractor = MLP(input_size, hidden_sizes, output_size=hidden_sizes[-1])
        self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def predict(self, x):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        return Normal(mean, std)
        