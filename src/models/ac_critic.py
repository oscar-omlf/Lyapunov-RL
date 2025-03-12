import torch
import torch.nn as nn

from models.mlp import MLP


class ACCritic(nn.Module):
    def __init__(self, input_size, hidden_sizes=(64, 64)):
        super(ACCritic, self).__init__()
        self.model = MLP(input_size, hidden_sizes, output_size=1)

    def forward(self, x):
        return self.model(x)
