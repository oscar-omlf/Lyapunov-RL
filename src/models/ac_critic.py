import torch
import torch.nn as nn


class ACCritic(nn.Module):
    def __init__(self, input_size, hidden_sizes=(64,64)):
        super(ACCritic, self).__init__()
        self.dims = [input_size] + list(hidden_sizes) + [1]
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
