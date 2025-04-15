#!/usr/bin/env python3
"""
This script is a converted version of the original Jupyter notebook.
It only keeps the Inverted Pendulum benchmark. All benchmarks for the 
Double Integrator, VanderPol, and Bicycle Tracking environments have been removed.
Also, all direct dreal imports and function calls are commented out (or replaced with dummy returns)
so that the script does not require dreal to run.
"""

import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Utility functions for converting between torch and numpy
def torch_to_np(x):
    return x.cpu().detach().numpy()

def np_to_torch(x):
    return torch.from_numpy(x).float().to(device)

def pipe(x, *funcs):
    for f in funcs:
        x = f(x)
    return x

# =============================================================================
# dreal-related utilities (dreal import and calls are commented out)
# =============================================================================

# import dreal as d  # <-- Commented out as requested

def dreal_var(n, name='x'):
    # return np.array([d.Variable("%s%d" % (name, i)) for i in range(n)])
    return np.zeros(n)  # dummy replacement

def dreal_elementwise(x, func):
    # return np.array([func(x[i]) for i in range(len(x))])
    return x

def dreal_sigmoid(x):
    # return 1 / (1 + d.exp(-x))
    return 1 / (1 + np.exp(-x))

# =============================================================================
# Benchmark base class and Inverted Pendulum (only environment kept)
# =============================================================================

class Benchmark():
    def __init__(self) -> None:
        self.name = None
        self.nx = None
        self.nu = None
        self.lb = None
        self.ub = None
        self.init_control = None  # [nu, nx]
    
    def f_np(self, x, u):
        pass

    def f_torch(self, x, u):
        pass

    def f_dreal(self, x, u):
        pass

    def in_domain_dreal(self, x, scale=1.):
        # return d.And(
        #     x[0] >= self.lb[0] * scale,
        #     x[0] <= self.ub[0] * scale,
        #     x[1] >= self.lb[1] * scale,
        #     x[1] <= self.ub[1] * scale
        # )
        return True

    def on_boundry_dreal(self, x, scale=2.):
        # condition1 = d.And(
        #     x[0] >= self.lb[0] * scale * 0.99,
        #     x[0] <= self.ub[0] * scale * 0.99,
        #     x[1] >= self.lb[1] * scale * 0.99,
        #     x[1] <= self.ub[1] * scale * 0.99
        # )
        # condition2 = d.Not(
        #     d.And(
        #         x[0] >= self.lb[0] * scale * 0.97,
        #         x[0] <= self.ub[0] * scale * 0.97,
        #         x[1] >= self.lb[1] * scale * 0.97,
        #         x[1] <= self.ub[1] * scale * 0.97
        #     )
        # )
        # return d.And(condition1, condition2)
        return True

    def sample_in_domain(self, n, scale=1.):
        pass

    def sample_out_of_domain(self, n, scale=1.):
        pass

class InvertedPendulum(Benchmark):
    def __init__(self):
        super().__init__()
        self.name = 'pendulum'
        self.nx = 2
        self.nu = 1
        self.lb = np.array([-2., -4.])
        self.ub = np.array([2., 4.])
        self.init_control = np.array([[-23.58639732,  -5.31421063]])
        
        self.G = 9.81   # gravity
        self.L = 0.5    # length of the pole
        self.m = 0.15   # ball mass
        self.b = 0.1    # friction
        
    def f_np(self, x, u):
        theta = x[:, 0]
        thetad = x[:, 1]
        thetadd = self.G / self.L * np.sin(theta) - self.b / (self.m * self.L**2) * thetad \
                  + u[:, 0] / (self.m * self.L**2)
        return np.stack([thetad, thetadd], axis=1)
    
    def f_torch(self, x, u):
        theta = x[:, 0]
        thetad = x[:, 1]
        thetadd = self.G / self.L * torch.sin(theta) - self.b / (self.m * self.L**2) * thetad \
                  + u[:, 0] / (self.m * self.L**2)
        return torch.stack([thetad, thetadd], dim=1)
    
    def f_dreal(self, x, u):
        # theta = x[0]
        # thetad = x[1]
        # thetadd = self.G / self.L * d.sin(theta) - self.b / (self.m * self.L**2) * thetad + u[0] / (self.m * self.L**2)
        # return np.array([thetad, thetadd])
        return np.array([0, 0])
        
    def sample_in_domain(self, n, scale=1.):
        return np.random.uniform(self.lb * scale, self.ub * scale, (n, self.nx))
    
    def sample_out_of_domain(self, n, scale=1.):
        x = np.random.uniform(-1, 1, (n, self.nx))
        xnorm = np.maximum(np.abs(x[:, 0]) / (self.ub[0] * scale), 
                           np.abs(x[:, 1]) / (self.ub[1] * scale))
        x = x / xnorm[:, None]
        noise = np.random.uniform(0, 0.5, (n, self.nx))
        x = x + np.sign(x) * noise
        return x

# =============================================================================
# Neural network used for the controller and value function approximation
# =============================================================================

class TanhNetwork(nn.Module):
    def __init__(self, dims, final_act='tanh'):
        super().__init__()
        self.dims = dims
        self.final_act = final_act
        
        layers = []
        for i in range(len(dims)-2):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if final_act == 'tanh':
            layers.append(nn.Tanh())
        elif final_act == 'sigmoid':
            layers.append(nn.Sigmoid())
        else:
            raise Exception("Not Implemented")
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x).squeeze(-1)
    
    def forward_with_grad(self, x):
        assert self.dims[-1] == 1
        y = self(x)
        jacob = torch.autograd.functional.jacobian(self, (x,), create_graph=True)
        jacob = jacob[0]
        grad = torch.diagonal(jacob).T
        return y, grad
    
    def get_param_pair(self):
        ws = []
        bs = []
        for name, param in self.layers.named_parameters():
            if "weight" in name:
                ws.append(param.cpu().detach().numpy())
            elif "bias" in name:
                bs.append(param.cpu().detach().numpy())
        if len(bs) == 0:
            bs = [np.zeros([w.shape[0]]) for w in ws]
        return ws, bs
    
    def forward_dreal(self, x):
        ws, bs = self.get_param_pair()
        # for w, b in zip(ws[:-1], bs[:-1]):
        #     x = dreal_elementwise(w @ x + b, d.tanh)
        # x = ws[-1] @ x + bs[-1]
        # if self.final_act == 'tanh':
        #     x = dreal_elementwise(x, d.tanh)
        # elif self.final_act == 'sigmoid':
        #     x = dreal_elementwise(x, dreal_sigmoid)
        # return x
        return np.array([0])

# =============================================================================
# Trainer class that handles training and simulation for the system
# =============================================================================

def rk4_step(f, x, u, dt):
    """ Runge-Kutta 4 integration step for a continuous-time system """
    f1 = f(x, u)
    f2 = f(x + 0.5 * dt * f1, u)
    f3 = f(x + 0.5 * dt * f2, u)
    f4 = f(x + dt * f3, u)
    return x + (dt / 6.0) * (f1 + 2 * f2 + 2 * f3 + f4)

class Trainer():
    def __init__(self,
                 system: Benchmark,
                 w_dim,
                 c_dim, 
                 alpha=0.01,
                 batch_size=16,
                 path_sampled=8,
                 integ_threshold=200,
                 norm_threshold=1e-2,
                 dt=0.05,
                 learning_rate=1e-2):
        # Create directories for saving results if they do not exist
        if not os.path.exists(f'./{system.name}'):
            os.mkdir(f'./{system.name}')
        if not os.path.exists(f'./{system.name}/plots'):
            os.mkdir(f'./{system.name}/plots')
        if not os.path.exists(f'./{system.name}/ckpts'):
            os.mkdir(f'./{system.name}/ckpts')
        
        self.system = system
        self.controller = TanhNetwork(c_dim, final_act='tanh')
        self.W = TanhNetwork(w_dim, final_act='sigmoid')
        self.alpha = alpha
        self.batch_size = batch_size
        self.path_sampled = path_sampled
        self.integ_threshold = integ_threshold
        self.norm_threshold = norm_threshold
        self.dt = dt
        self.learning_rate = learning_rate
        self.zero = np_to_torch(np.zeros([1, system.nx]))
        
    def load_ckpt(self, fname):
        ckpt = torch.load(fname, map_location=device)
        self.W.load_state_dict(ckpt['W'])
        self.controller.load_state_dict(ckpt['C'])
        
    @torch.no_grad()
    def simulate_trajectory(self, x=None, max_steps=int(1e7)):
        integ_acc = 0.
        steps = 0
        x = np_to_torch(self.system.sample_in_domain(1)) if x is None else x
        x_hist = [x.clone()]
        
        while True:
            steps += 1
            norm = torch.linalg.norm(x).item()
            integ_acc += norm * self.dt
            
            if norm < self.norm_threshold or (len(x_hist) > 10 and torch.linalg.norm(x_hist[-1] - x_hist[-10]) < 1e-3) or steps >= max_steps:
                return torch.cat(x_hist, dim=0), integ_acc, True
            elif integ_acc > self.integ_threshold:
                return torch.cat(x_hist, dim=0), integ_acc, False
            
            u = self.controller(x)
            if self.system.nu == 1:
                u = u[:, None]
            x = rk4_step(self.system.f_torch, x_hist[-1], u, dt=self.dt)
            x_hist.append(x.clone())
            
    def train(self, iterations=2000):
        num_iter = 0
        optimizer = torch.optim.Adam(list(self.W.parameters()) + list(self.controller.parameters()),
                                     lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=500, gamma=0.8)
        
        # Prepare a grid for plotting (only for 2D data)
        xx_plot = np.linspace(self.system.lb[0] * 2, self.system.ub[0] * 2, 3000)
        yy_plot = np.linspace(self.system.lb[1] * 2, self.system.ub[1] * 2, 3000)
        X_plot, Y_plot = np.meshgrid(xx_plot, yy_plot)
        xys_plot = np.stack([X_plot, Y_plot], axis=-1).reshape([-1, 2])
        xys_plot = np_to_torch(xys_plot)
        
        for unused in range(iterations):
            num_iter += 1
            vs = []
            xs = []
            
            for _ in range(self.path_sampled):
                traj, integ, _ = self.simulate_trajectory()
                xs.append(traj[0])
                vs.append(integ)
            xs = torch.stack(xs, dim=0)
            vs = np_to_torch(np.array(vs))
            
            critic_loss = 0.
            actor_loss = 0.
            
            # W(0) = 0
            W0 = self.W(self.zero)
            critic_loss += 5. * torch.square(W0)
            
            # Approximate W(x) = tanh(alpha * V(x))
            Wx = self.W(xs)
            What = torch.tanh(self.alpha * vs)
            critic_loss += F.mse_loss(Wx, What)
            
            # PDE residual: Physics-Informed Loss
            xs = np_to_torch(self.system.sample_in_domain(self.batch_size, scale=1.))
            Wx, grad_Wx = self.W.forward_with_grad(xs)
            us = self.controller(xs)
            if self.system.nu == 1:
                us = us[:, None]
            fx = self.system.f_torch(xs, us)
            xnorm = torch.norm(xs, p=2, dim=1)
            residual = torch.sum(grad_Wx * fx.detach(), dim=1) + self.alpha * xnorm * (1 + Wx) * (1 - Wx)
            critic_loss += torch.mean(torch.square(residual))
            
            # Controller loss
            grad_Wx = grad_Wx / torch.linalg.norm(grad_Wx, dim=1, keepdim=True)
            actor_loss += torch.mean(torch.sum(grad_Wx.detach() * fx, dim=1))
            
            # Barrier loss on out-of-domain samples
            xs = np_to_torch(self.system.sample_out_of_domain(self.batch_size, scale=2.))
            Wx = self.W(xs)
            critic_loss += 2. * F.l1_loss(Wx, torch.ones_like(Wx).to(device))
                
            loss = 0.5 * actor_loss + critic_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if num_iter % 10 == 0:
                print(num_iter, critic_loss.item(), actor_loss.item())
            
            if num_iter % 20 == 0:
                vals = pipe(xys_plot, self.W, torch_to_np)
                fig = plt.figure()
                ax = plt.axes()
                im = ax.contourf(X_plot, Y_plot, np.reshape(vals, [len(xx_plot), len(yy_plot)]), levels=100)
                fig.colorbar(im)
                
                for _ in range(5):
                    path, _, convergence = self.simulate_trajectory(max_steps=3000)
                    if convergence:
                        path = torch_to_np(path)
                        plt.plot(path[0, 0], path[0, 1], 'r+')
                        plt.plot(path[:, 0], path[:, 1], 'o-', markersize=1, linewidth=0.5)
                plt.gca().set_aspect('equal')
                plt.savefig(f'./{self.system.name}/plots/{num_iter}.png')
                plt.close()
                
            if num_iter % 1000 == 0:
                torch.save({
                    'W': self.W.state_dict(),
                    'C': self.controller.state_dict()
                }, f'./{self.system.name}/ckpts/{num_iter}.pth')
                
    def check_lyapunov(self, level=0.9, scale=2., eps=0.5):
        # The original implementation used dreal expressions.
        # This dummy implementation always returns (True, True).
        return True, True

# =============================================================================
# Main function to run training for the Inverted Pendulum
# =============================================================================

if __name__ == "__main__":
    # Create the inverted pendulum system and corresponding trainer
    system = InvertedPendulum()
    trainer = Trainer(system,
                      w_dim=[system.nx, 20, 20, 1],
                      c_dim=[system.nx, 5, 5, system.nu],
                      alpha=0.2,
                      dt=0.003,
                      norm_threshold=5e-2,
                      integ_threshold=150,
                      batch_size=64,
                      path_sampled=8,
                      learning_rate=2e-3)
    
    # Start training (change the iteration count as needed)
    trainer.train(3000)
    
    # After training, perform a sample forward pass and print some values.
    x = np_to_torch(np.array([[-0.083125, 1.]]))
    Wx, gradWx = trainer.W.forward_with_grad(x)
    u = trainer.controller(x)[:, None]
    fx = trainer.system.f_torch(x, u)
    
    print("W(x):", Wx)
    print("Inner product <grad W, f(x)>:", torch.sum(gradWx * fx, dim=1))
    print("W(0):", trainer.W(trainer.zero))
