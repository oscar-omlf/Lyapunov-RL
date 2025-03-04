import torch
import torch.nn as nn

class LyapunovCritic(nn.Module):
    """
    This network W_\theta(x) is the Zubov function approximation, in range (0,1).
    We'll do a final Sigmoid to keep it in (0,1).
    """
    def __init__(self, input_size, hidden_sizes=(64,64)):
        super(LyapunovCritic, self).__init__()
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

    def forward(self, x):
        return self.layers(x).squeeze(-1)
    
    def forward_with_grad(self, x):
        """
        Compute both the network output and its gradient with respect to input x.
        :param x: Tensor of shape [batch, input_size]
        :return:
            y: Tensor of shape [batch] (scalar per sample)
            grad: Tensor of shape [batch, input_size], where grad[i] = d/dx W(x_i)
        """
        # Ensure the network's output dimension is 1.
        assert self.dims[-1] == 1
        y = self(x)  # Shape: [batch]
        
        # Compute the Jacobian of the network output with respect to x.
        jacob = torch.autograd.functional.jacobian(self, (x,), create_graph=True)[0]
        # jacob has shape [batch, batch, input_size].
        # Extract the diagonal: for each sample i, grad[i] = d/dx W(x_i)
        grad = torch.diagonal(jacob, dim1=0, dim2=1).T  # Resulting shape: [batch, input_size]
        return y, grad
