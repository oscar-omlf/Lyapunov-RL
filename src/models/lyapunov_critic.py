import torch
import torch.nn as nn

from models.mlp import MLP

class LyapunovCritic(nn.Module):
    def __init__(self, input_size, hidden_sizes=(64,64)):
        super(LyapunovCritic, self).__init__()
        self.model = MLP(input_size, hidden_sizes, output_size=1, output_activation=nn.Sigmoid())

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def forward_with_grad(self, x):
        # Compute the network output.
        y = self(x)
        # Compute the Jacobian with respect to input x.
        jacob = torch.autograd.functional.jacobian(self, (x,), create_graph=True)[0]
        grad = torch.diagonal(jacob, dim1=0, dim2=1).T
        return y, grad