import torch.nn as nn

class LyapunovCritic(nn.Module):
    """
    This network W_\theta(x) is the Zubov function approximation, in range (0,1).
    We'll do a final Sigmoid to keep it in (0,1).
    """
    def __init__(self, input_size, hidden_sizes=(64,64)):
        super(LyapunovCritic, self).__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)