import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from trainers.abstract_trainer import AbstractTrainer

class ACTrainer(AbstractTrainer):
    def __init__(self, buffer, actor, critic, gamma, n_steps, actor_lr, critic_lr, device):
        """
        Initialize the ACTrainer with the replay buffer, actor, critic,
        discount factor, number of steps for n-step return, learning rates, and device.
        """
        self.buffer = buffer
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_loss_fn = nn.MSELoss()

    def train(self):
        """
        Train the actor and critic using all transitions in the replay buffer.
        Assumes the buffer contains transitions from a single episode.
        Each transition is a tuple: (state, action, reward, next_state).
        Computes n-step returns for each time step, then updates both networks.
        """
        if len(self.buffer) == 0:
            return None

        # Retrieve and clear all transitions from the buffer.
        # Note: We assume transitions were collected in order for a single episode.
        transitions = list(self.buffer.buffer)
        self.buffer.buffer.clear()

        states, actions, targets, advantages = [], [], [], []
        L = len(transitions)

        for t in range(L):
            state_t, action_t, reward_t, _ = transitions[t]
            reward_t = float(reward_t)

            # Compute n-step return G_t.
            G = 0.0
            discount = 1.0
            n = min(self.n_steps, L - t)
            for k in range(n):
                _, _, r, _ = transitions[t + k]
                G += discount * float(r)
                discount *= self.gamma

            # If there is a next state available beyond the n-step window, add bootstrap.
            if t + self.n_steps < L:
                state_tpn, _, _, _ = transitions[t + self.n_steps]
                state_tpn_tensor = torch.tensor(state_tpn, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    bootstrap_value = self.critic(state_tpn_tensor).item()
                G += discount * bootstrap_value

            # Compute the critic's value for state_t.
            state_t_tensor = torch.tensor(state_t, dtype=torch.float32, device=self.device).unsqueeze(0)
            value_t = self.critic(state_t_tensor).item()
            advantage = G - value_t

            states.append(state_t)
            actions.append(action_t)
            targets.append(G)
            advantages.append(advantage)

        # Convert lists to tensors.
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
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
        entropy_coef = 0.01
        actor_loss -= entropy_coef * entropy

        total_loss = actor_loss + critic_loss

        # Backpropagation and parameter update.
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return total_loss.item()
