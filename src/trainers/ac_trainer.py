import torch
from torch import nn
from agents.abstract_agent import ReplayBuffer

from models.twoheadedmlp import TwoHeadedMLP
from trainers.abstract_trainer import Trainer
from util.device import fetch_device


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

        print(f'Trajectory check!! {trajectory == trajectory_1}') # hopefully True

        return trajectory

    def _do_single_update(self, transitions) -> tuple[float, float]:
        """
        Perform exactly ONE update (n-step or partial) for the EARLIEST transition in 'transitions'.
        Returns (actor_loss, critic_loss).

        Steps:
          1) Label the earliest transition as (s0,a0,...) = transitions[0]
          2) Sum rewards for up to n steps or until done => G
          3) If no 'done' encountered, bootstrap from V(s_{last_next})
          4) advantage = G - V(s0)
          5) Update critic, update actor
          6) popleft() that earliest transition from the buffer
        """

        s0, a0, r0, s1, done0 = transitions[0]  

        # Compute the return G from the earliest transition forward
        G = 0.0
        discount = 1.0
        done_encountered = False
        n_collected = 0
        for i, (s_i, a_i, r_i, s_i1, d_i) in enumerate(transitions):
            if i >= self.n_steps:
                break  # we only look at up to n steps
            G += discount * r_i.item()
            discount *= self.gamma
            n_collected += 1
            if d_i.item() == 1:  # done encountered
                done_encountered = True
                break

        # 3) If we didn't encounter done, we bootstrap from the last 'next_state'
        if not done_encountered:
            # the last transition we used was transitions[n_collected-1]
            _, _, _, last_next_state, _ = transitions[n_collected-1]
            with torch.no_grad():
                G += discount * self.critic_model(last_next_state).item()

        # 4) advantage = G - V(s0)
        G_tens = torch.tensor([G], dtype=torch.double, device=self.device)
        value_s0 = self.critic_model(s0)  # shape [1]
        advantage = G_tens - value_s0

        # 5) Update critic
        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        # We call .predict(...) from your MLPMultivariateGaussian
        dist = self.actor_model.predict(s0)  # returns a MultivariateNormal
        log_prob_a0 = dist.log_prob(a0)
        actor_loss = -log_prob_a0 * advantage.detach()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 6) Remove the earliest transition from the buffer (so we don't update it again)
        self.buffer.buffer.popleft()

        return (actor_loss.item(), critic_loss.item())

    def train(self):
        """
        Called after each step (or at the end of the episode as well).
        - If we have >= n_steps transitions and the last transition is NOT done, do exactly ONE n-step update.
        - Else if the last transition is done => we 'drain' leftover transitions:
            repeatedly do partial updates (one transition at a time) until the buffer is empty.
        - Return (actor_loss, critic_loss) for the *last* update performed,
          or None if no update was done.
        """
        transitions = self._trajectory()
        if len(transitions) == 0:
            return None  # nothing to do

        # check the last transition
        *_, last_done = transitions[-1]

        # 1) CASE: if we have enough transitions (>= n_steps) and last is not done => do ONE n-step update
        if (len(transitions) >= self.n_steps) and (last_done.item() == 0):
            # standard n-step update for the earliest transition
            return self._do_single_update(transitions)

        # 2) CASE: if last_done = True, we know episode ended => we 'drain' leftover transitions
        if last_done.item() == 1:
            # We'll do repeated partial updates until the buffer is empty
            # or until we can't do more (but typically we do until empty).
            last_loss = None
            while len(self.buffer) > 0:
                transitions = self._trajectory()
                # do a single partial update for the earliest
                last_loss = self._do_single_update(transitions)
                # we keep going until buffer is empty
            return last_loss

        # 3) CASE: if we have fewer than n_steps transitions and the last is not done => do nothing
        return None
    