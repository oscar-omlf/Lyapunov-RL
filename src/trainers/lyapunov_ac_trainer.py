import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.twoheadedmlp import TwoHeadedMLP
from models.sampling import sample_two_headed_gaussian_model
from trainers.abstract_trainer import Trainer


class LyapunovACTrainer(Trainer):
    def __init__(
        self, 
        actor: TwoHeadedMLP, 
        critic: nn.Module,
        actor_lr: float, 
        critic_lr: float,
        alpha: float,
        batch_size: int,
        num_paths_sampled: int,
        state_space: int,
        r1_bounds: list,
        device: str
    ):
        super().__init__()
        self.actor_model = actor
        self.critic_model = critic

        self.alpha = alpha
        self.batch_size = batch_size
        self.num_paths_sampled = num_paths_sampled

        self.state_space = state_space
        self.lb, self.ub = r1_bounds

        self.device = device

        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    def train(self, iterations: int = 1000):
        curr_iter = 0

        for _ in range(iterations):
            init_states = []
            values = []

            for _ in range(self.num_paths_sampled):
                traj, value = self.simulate_trajectory()
                init_states.append(traj[0])
                values.append(value)

            init_states = torch.stack(init_states)
            values = torch.as_tensor(values, dtype=torch.float32, device=self.device)

            actor_loss = 0
            critic_loss = 0

            # 1) Enforces W(0) = 0
            zeros_tensor = torch.zeros((1, self.state_space), dtype=torch.float32, device=self.device)
            Lz = self.critic_model(zeros_tensor)

            # 2) Enforces W(x) = tanh(alpha * V(x))
            Wx = self.critic_model(init_states)
            target = torch.tanh(self.alpha * values)
            Lr = F.mse_loss(Wx, target)

            # 3) Physics-Informed Loss (PDE residual)
            init_states = sample_in_region(self.batch_size, self.lb, self.ub)  # Sample from R1
            init_states = torch.as_tensor(init_states, dtype=torch.float32, device=self.device)

            Wx, grad_Wx = self.critic_model.forward_with_grad(init_states)
                        
            us, _ = sample_two_headed_gaussian_model(self.actor_model, init_states)
            fxu = f_torch(init_states, us)

            phix = torch.linalg.vector_norm(init_states, ord=2, dim=1)
            resid = torch.sum(grad_Wx * fxu.detach(), dim=1) + self.alpha * (1 + Wx) * (1 - Wx) * phix
            Lp = torch.mean(torch.square(resid))

            # 4) Encourage control actions that decrease the Lyapunov function.
            grad_norm = torch.linalg.vector_norm(grad_Wx, ord=2, dim=1, keepdim=True)
            unit_grad = grad_Wx / (grad_norm + 1e-8)
            Lc = torch.mean(torch.sum(unit_grad.detach() * fxu, dim=1))

            # 5) Enforce that on the boundary of R2, W(x) ≈ 1
            init_states = sample_out_of_region(self.batch_size, self.lb, self.ub, scale=2)  # Sample from R2
            init_states = torch.as_tensor(init_states, dtype=torch.float32, device=self.device)
            Wx = self.critic_model(init_states)
            Lb = torch.mean(torch.abs(Wx - 1.0))

            actor_loss = Lc
            critic_loss = Lz + Lr + Lp + Lb

            # Update the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


    def simulate_trajectory(self, x = None, max_steps = int(1e7)):
        pass
        

def sample_in_region(num_samples: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    return np.random.uniform(lb, ub, size=(num_samples, lb.shape[0]))

def sample_out_of_region(num_samples: int, lb: np.ndarray, ub: np.ndarray, scale: float = 2.0) -> np.ndarray:
    x = np.random.uniform(-1, 1, size=(num_samples, lb.shape[0]))
    # For each sample, compute the maximum ratio: |x_i| / (ub_i * scale)
    ratios = np.max(np.abs(x) / (ub * scale), axis=1, keepdims=True)
    # Scale the sample so that at least one coordinate is at the boundary
    x = x / ratios
    # Add small noise in the same sign direction to push samples outside
    noise = np.random.uniform(0, 0.5, size=x.shape)
    x = x + np.sign(x) * noise
    return x

def f_torch(xs, us, dt=0.05):
    """
    Differentiable dynamics function for Gymnasium's Pendulum-v1.
    Assumes xs has shape (batch, 3): [cos(theta), sin(theta), theta_dot],
    and us has shape (batch, 1).
    Returns dx/dt with the same shape as xs.
    """
    # Recover theta from [cos(theta), sin(theta)]
    theta = torch.atan2(xs[:, 1], xs[:, 0])
    theta_dot = xs[:, 2]
    
    # For Gymnasium's Pendulum-v1 (https://gymnasium.farama.org/environments/classic_control/pendulum/):
    g = 10.0  # gravitational acceleration
    m = 1.0   # mass
    l = 1.0   # pendulum length
    
    # Gymnasium's dynamics update for angular acceleration:
    # dtheta_dot/dt = 3 * g / (2 * l) * sin(theta) + 3.0 / (m * l**2) * u
    theta_ddot = (3 * g / (2 * l)) * torch.sin(theta) + (3.0 / (m * l**2)) * us.squeeze()
    
    # Compute derivatives for the trigonometric state components:
    # d(cos(theta))/dt = -sin(theta) * theta_dot
    dcos = -torch.sin(theta) * theta_dot
    # d(sin(theta))/dt = cos(theta) * theta_dot
    dsin = torch.cos(theta) * theta_dot
    # d(theta_dot)/dt = theta_ddot
    dtheta_dot = theta_ddot
    
    # For a continuous-time derivative (dx/dt)
    dxdt = torch.stack([dcos, dsin, dtheta_dot], dim=1)
    return dxdt


def train(trainer):
    trainer.train()
