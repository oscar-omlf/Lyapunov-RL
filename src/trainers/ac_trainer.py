import torch
from torch import nn
import torch.nn.functional as F

from agents.abstract_agent import ReplayBuffer
from models.twoheadedmlp import TwoHeadedMLP
from trainers.abstract_trainer import Trainer


class ACTrainer(Trainer):
    def __init__(
        self, 
        buffer: ReplayBuffer, 
        actor: TwoHeadedMLP, 
        critic: nn.Module,
        gamma: float, 
        n_steps: int, 
        actor_lr: float, 
        critic_lr: float, 
        device: str
    ):
        super().__init__()
        self.buffer = buffer
        self.actor_model = actor
        self.critic_model = critic
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    def train(self):
        trajectory = self.buffer.get_buffer_list()
        self.buffer.clear()
        if not trajectory:
            return None

        # Convert transitions to tensors
        states = torch.stack([t[0] for t in trajectory])  # (T, state_dim)
        actions = torch.stack([t[1] for t in trajectory])  # (T,)
        rewards = torch.stack([t[2] for t in trajectory])  # (T,)
        next_states = torch.stack([t[3] for t in trajectory])  # (T, state_dim)
        
        T = len(trajectory)
        returns = []

        # Calculate N-step returns for each timestep (Algorithm lines 4-6)
        for t in range(T):
            sum_rewards = 0.0
            for k in range(self.n_steps):
                if t + k < T:
                    sum_rewards += (self.gamma ** k) * rewards[t + k]
            
            # Calculate V_end term (s_{t+N} if valid)
            if t + self.n_steps < T:
                s_t_plus_n = states[t + self.n_steps]
                v_end = self.critic_model(s_t_plus_n).squeeze().detach()
            else:
                v_end = 0.0
                
            R_t = sum_rewards + (self.gamma ** self.n_steps) * v_end
            returns.append(R_t)

        returns_tensor = torch.stack(returns)  # (T,)

        # Calculate advantages (R_t - V(s_t))
        current_values = self.critic_model(states).squeeze()  # (T,)
        advantages = returns_tensor - current_values.detach()  # (T,)

        # Policy loss calculation (Algorithm line 7)
        dist = self.actor_model.predict(states)  # MultivariateNormal
        # Reshape actions to (T, action_dim=1) for log_prob
        log_probs = dist.log_prob(actions)  # (T,)
        actor_loss = -(advantages * log_probs).mean()

        # Value loss calculation (Algorithm line 8)
        critic_loss = 0.5 * (returns_tensor - current_values).pow(2).mean()

        # Update networks (Algorithm lines 9-10)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()
