import numpy as np
import torch.nn as nn
import dreal as d

from util.dreal import dreal_var, dreal_elementwise, dreal_sigmoid


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, inner_activation=nn.ReLU(), output_activation=None):
        """
        :param inner_activation: activation function to use for all hidden layers.
        """
        super(MLP, self).__init__()
        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(inner_activation)
            last_size = size
        layers.append(nn.Linear(last_size, output_size))
        if output_activation is not None:
            layers.append(output_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def get_param_pair(self):
        """Extracts weights and biases for each Linear layer in order."""
        ws = []
        bs = []
        for module in self.net:
            if isinstance(module, nn.Linear):
                ws.append(module.weight.detach().cpu().numpy())
                bs.append(module.bias.detach().cpu().numpy())
        return ws, bs

    def forward_dreal(self, x):
        """
        Construct a dReal symbolic expression for the MLP.
        `x` is expected to be a numpy array of dReal Variables.
        """
        out = x
        for module in self.net:
            if isinstance(module, nn.Linear):
                W = module.weight.detach().cpu().numpy()
                b = module.bias.detach().cpu().numpy()
                # Compute the affine transformation: W*out + b.
                # Here we assume `out` is a numpy array of dReal variables.
                out = np.dot(W, out) + b
            elif isinstance(module, nn.Tanh):
                out = dreal_elementwise(out, d.tanh)
            elif isinstance(module, nn.Sigmoid):
                out = dreal_elementwise(out, dreal_sigmoid)
            else:
                raise NotImplementedError("Activation not supported in dReal forward pass")
        return out
