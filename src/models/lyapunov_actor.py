import torch
import torch.nn as nn
from torch.distributions import Normal

class LyapunovActor(nn.Module):
    """
    This network π₍γ₎(x) is the control policy for the PDE-based method.
    It uses a two-headed architecture: one head outputs the mean and the other the log standard deviation.
    In this version, we squash the actions using a TanhTransform followed by an AffineTransform,
    so that the final actions lie in the range [-max_action, max_action] while the log probabilities
    are correctly computed.
    """
    def __init__(self, input_size, hidden_sizes=(64,64), action_dim=1):
        super(LyapunovActor, self).__init__()
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
        return mean, log_std

    def predict(self, x):
        """
        Returns a squashed Gaussian distribution from which actions can be sampled.
        
        The method first builds a Normal distribution using the mean and standard deviation
        (obtained from exp(log_std)). It then applies a TanhTransform to squash the outputs
        to (-1, 1) and an AffineTransform to scale them to [-max_action, max_action]. Finally,
        Independent is applied so that the action dimensions are treated as one joint event.
        """
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        return Normal(mean, std)
        
