import torch
from torch import nn
import torch.nn.functional as F

from agents.abstract_agent import ReplayBuffer
from models.twoheadedmlp import TwoHeadedMLP
from trainers.abstract_trainer import Trainer

from models.sampling import log_prob_policy


class ACTrainer(Trainer):
    def __init__(
        self, 
        buffer: ReplayBuffer, 
        actor: TwoHeadedMLP, 
        critic: nn.Module,
        gamma: float, 
        n_steps: int, 
        batch_size: int,
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
        self.batch_size = batch_size
        self.device = device

        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    def train(self):
        transitions = self.buffer.get_buffer_list()
        T = len(transitions)
        if T == 0: return None

        # Extract tensors
        states = torch.stack([t[0] for t in transitions])
        actions = torch.stack([t[1] for t in transitions])
        rewards = torch.stack([t[2] for t in transitions])
        next_states = torch.stack([t[3] for t in transitions])
        dones = torch.stack([t[4] for t in transitions])

        # Compute V(s_t)
        V_values = self.critic_model(states).squeeze()

        # Compute N-step returns with done handling
        R = torch.zeros(T, device=self.device)
        for t in range(T):
            sum_rewards = 0.0
            gamma_k = 1.0  # γ^0
            for k in range(self.n_steps):
                if t + k >= T:
                    break
                sum_rewards += gamma_k * rewards[t + k]
                gamma_k *= self.gamma  # Update γ^k

                if dones[t + k]:
                    break

            # Compute V_end (0 if terminated within N steps)
            if t + k < T and not dones[t + k]:
                s_t_plus_k = next_states[t + k]
                with torch.no_grad():
                    V_end = self.critic_model(s_t_plus_k.unsqueeze(0)).squeeze()
            else:
                V_end = 0.0

            R[t] = sum_rewards + gamma_k * V_end

        # Compute losses using your existing functions
        advantages = R - V_values.detach()
        log_probs = log_prob_policy(self.actor_model, states, actions)
        actor_loss = -(log_probs * advantages).sum()
        critic_loss = 0.5 * torch.sum((R - V_values) ** 2)

        # Optimize
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.buffer.clear()

        return actor_loss.item(), critic_loss.item()

