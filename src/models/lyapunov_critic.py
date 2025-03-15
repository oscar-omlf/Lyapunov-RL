import torch
import torch.nn as nn

from models.mlp import MLP

class LyapunovCritic(nn.Module):
    def __init__(self, input_size, hidden_sizes=(20, 20), inner_activation=nn.Tanh()):
        super(LyapunovCritic, self).__init__()
        self.model = MLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            inner_activation=inner_activation,
            output_activation=nn.Sigmoid()
            )

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def forward_with_grad(self, x):
        y = self(x)
        jacob = torch.autograd.functional.jacobian(self, (x,), create_graph=True)[0]
        grad = torch.diagonal(jacob, dim1=0, dim2=1).T
        return y, grad
    
    def forward_dreal(self, x):
        """Construct dReal symbolic expression for the critic."""
        return self.model.forward_dreal(x)