import torch
from torch import nn
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
        critic_lr: float
    ):
        """
        Initialize the Actor-Critic Trainer.
        """
        self.buffer = buffer
        self.actor_model = actor
        self.critic_model = critic
        self.gamma = gamma
        self.n_steps = n_steps

        self.actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=critic_lr)

    def _trajectory(self) -> tuple:
        """
        Sample the trajectory from the replay buffer.
        """
        trajectory = self.buffer.get_buffer_list()
        try:
            trajectory_1 = self.buffer.get_buffer_list()[-self.n_steps:]
        except IndexError:
            print('IndexError: n_steps is greater than buffer length')

        if not trajectory == trajectory_1:
            print(f'Trajectory check!! {trajectory == trajectory_1}') # hopefully True

        return trajectory

    def train(self, flush: bool = False):
        """
        Perform exactly ONE n-step update (or a partial n-step update if flush is True)
        for the earliest transition(s) in the current buffer.
        
        If flush is False, the update only occurs if there are at least n_steps transitions.
        If flush is True, even if there are fewer than n_steps transitions,
        a partial update is performed and the remaining transitions are then flushed.
        
        Returns (actor_loss, critic_loss) or None if no update occurred.
        """
        transitions = self._trajectory()
        total_transitions = len(transitions)

        if total_transitions == 0:
            return None
    
        if not flush and total_transitions < self.n_steps:
            return None

        # Use as many transitions as available (which is <= n_steps)
        n = min(self.n_steps, total_transitions)
        if self.n_steps < total_transitions:
            print('n_steps is less than buffer length')
        G = 0.0
        found_terminal = False
        for i in range(n):
            # Each transition: (state, action, reward, next_state, done)
            _, _, r, _, done = transitions[i]
            G += (self.gamma ** i) * r.item()  # r is a tensor; extract scalar via .item()
            if done.item():
                found_terminal = True
                n = i + 1  # Use only transitions up to terminal.
                break

        # If we have a full n-step batch and no terminal was found, bootstrap from critic.
        if not found_terminal and n == self.n_steps:
            s_bootstrap = transitions[n - 1][3]  # next_state tensor from the nth transition
            V_bootstrap = self.critic_model(s_bootstrap)
            G += (self.gamma ** n) * V_bootstrap.item()

        # Update on the earliest transition.
        s0, a0, _, _, _ = transitions[0]
        V_s0 = self.critic_model(s0)
        advantage = torch.tensor(G, dtype=V_s0.dtype, device=V_s0.device) - V_s0

        # Critic loss: MSE between target and V(s0)
        critic_loss = advantage.pow(2)

        # Actor loss for continuous actions:
        # The actor model (MLPMultivariateGaussian) returns a distribution via its predict() method.
        distribution = self.actor_model.predict(s0)
        log_prob = distribution.log_prob(a0)  # a0 is already a tensor
        actor_loss = -log_prob * advantage.detach()

        # Separate updates for actor and critic:
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Remove used transitions from the buffer.
        for _ in range(n):
            self.buffer.popleft()

        # If flush is enabled, clear the entire buffer.
        if flush:
            self.buffer.clear()

        return actor_loss.item(), critic_loss.item()