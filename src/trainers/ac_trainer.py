import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from trainers.abstract_trainer import AbstractTrainer

class ACTrainer(AbstractTrainer):
    def __init__(self, buffer, actor, critic, gamma, n_steps, entropy_coef, actor_lr, critic_lr, device):
        """
        Initialize the ACTrainer with the replay buffer, actor, critic,
        discount factor, number of steps for n-step return, learning rates, and device.
        """
        self.buffer = buffer
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.device = device

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_loss_fn = nn.MSELoss()

    def train(self):
        if len(self.buffer) == 0:
            return None

        transitions = list(self.buffer.buffer)
        self.buffer.buffer.clear()

        states = [t[0] for t in transitions]
        actions = [t[1] for t in transitions]

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pre_update_values = self.critic(states_tensor).squeeze().cpu().numpy()

        targets, advantages = [], []
        L = len(transitions)

        for t in range(L):
            state_t, action_t, reward_t, _ = transitions[t]
            reward_t = float(reward_t)

            G = 0.0
            discount = 1.0
            n = min(self.n_steps, L - t)
            for k in range(n):
                _, _, r, _ = transitions[t + k]
                G += discount * float(r)
                discount *= self.gamma

            if t + n < L:
                bootstrap_value = pre_update_values[t + n]
                G += discount * bootstrap_value

            value_t = pre_update_values[t]
            advantage = G - value_t

            targets.append(G)
            advantages.append(advantage)

        # Convert to tensors (now includes actions)
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)  # Now works
        targets_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device).unsqueeze(1)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Critic loss: Mean Squared Error between predicted value and n-step return.
        predicted_values = self.critic(states_tensor)
        critic_loss = self.critic_loss_fn(predicted_values, targets_tensor)

        # Actor loss: negative log probability weighted by advantage.
        # Obtain the action distribution from the actor.
        dist = self.actor.predict(states_tensor)
        log_probs = dist.log_prob(actions_tensor)
        # If action space is multidimensional, sum log probabilities.
        if log_probs.dim() > 1:
            log_probs = log_probs.sum(dim=-1, keepdim=True)
        actor_loss = - (log_probs * advantages_tensor).mean()

        # Optional entropy bonus to encourage exploration.
        entropy = dist.entropy().mean()
        actor_loss -= self.entropy_coef * entropy

        total_loss = actor_loss + critic_loss

        # Backpropagation and parameter update.
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()
