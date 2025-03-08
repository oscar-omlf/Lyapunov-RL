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
        actor_update_interval: int,
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

        self.actor_update_interval = actor_update_interval
        self._update_counter = 0

        self.actor_optimizer = torch.optim.RMSprop(actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.RMSprop(critic.parameters(), lr=critic_lr)

    def train(self):
        trajectory = self.buffer.get_buffer_list()
        self.buffer.clear()
        if not trajectory:
            return None

        states     = torch.stack([t[0] for t in trajectory])  # (T, state_dim)
        actions    = torch.stack([t[1] for t in trajectory])  # (T, action_dim)
        rewards    = torch.stack([t[2] for t in trajectory])  # (T,)
        next_states= torch.stack([t[3] for t in trajectory])  # (T, state_dim)
        dones      = torch.stack([t[4] for t in trajectory])  # (T,) of bool

        T = len(trajectory)
        returns = []

        for t in range(T):
            sum_rewards = 0.0
            done_encountered = False

            for k in range(self.n_steps):
                idx = t + k
                if idx >= T:
                    break
                sum_rewards += (self.gamma ** k) * rewards[idx]
                if dones[idx]:
                    done_encountered = True
                    break
            
            if (not done_encountered) and ((t + self.n_steps) < T):
                v_end = self.critic_model(states[t + self.n_steps]).squeeze().detach()
                sum_rewards += (self.gamma ** self.n_steps) * v_end

            returns.append(sum_rewards)

        returns_tensor = torch.stack(returns)  # shape (T,)

        current_values = self.critic_model(states).squeeze()  # shape (T,)
        advantages = returns_tensor - current_values.detach()

        critic_loss = 0.5 * (returns_tensor - current_values).pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self._update_counter % self.actor_update_interval == 0:
            dist = self.actor_model.predict(states) 
            log_probs = dist.log_prob(actions)        # (T,)

            actor_loss = -(advantages * log_probs).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        else:
            actor_loss = None

        self._update_counter += 1

        if actor_loss is not None:
            return actor_loss.item(), critic_loss.item()
        else:
            return None, critic_loss.item()
