import torch
import numpy as np

from agents.abstract_agent import AbstractAgent
from models.lyapunov_actor import LyapunovActor
from models.lyapunov_critic import LyapunovCritic
from trainers.lyapunov_trainer import LyapunovTrainer


class LyapunovAgent(AbstractAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        
        self.alpha = config.get("alpha")
        self.lr = config.get("lr")
        self.dynamics_fn = config.get("dynamics_fn")
        self.dynamics_fn_dreal = config.get("dynamics_fn_dreal")
        self.batch_size = config.get("batch_size")
        self.num_paths_sampled = config.get("num_paths_sampled")
        self.dt = config.get("dt")
        self.norm_threshold = config.get("norm_threshold")
        self.integ_threshold = config.get("integ_threshold")
        self.r1_bounds = config.get("r1_bounds")
        
        state_dim = self.state_space.shape[0]
        action_dim = self.action_space.shape[0]

        actor_hidden_sizes = config.get("actor_hidden_sizes")
        critic_hidden_sizes = config.get("critic_hidden_sizes")

        self.max_action = config.get("max_action")
        
        self.actor_model = LyapunovActor(state_dim, actor_hidden_sizes, action_dim, max_action=self.max_action).to(device=self.device)
        self.critic_model = LyapunovCritic(state_dim, critic_hidden_sizes).to(device=self.device)

        self.run_dir = config.get("run_dir")

        self.trainer = LyapunovTrainer(
            actor=self.actor_model,
            critic=self.critic_model,
            lr=self.lr,
            alpha=self.alpha,
            batch_size=self.batch_size,
            num_paths_sampled=self.num_paths_sampled,
            norm_threshold=self.norm_threshold,
            integ_threshold=self.integ_threshold,
            dt=self.dt,
            dynamics_fn=self.dynamics_fn,
            dynamics_fn_dreal=self.dynamics_fn_dreal,
            state_dim=state_dim,
            r1_bounds=self.r1_bounds,
            run_dir=self.run_dir,
            device=self.device,
        )

    def add_transition(self, transition: tuple) -> None:
        pass

    def update(self, counter_examples: list = None):
        loss = self.trainer.train(counter_examples=counter_examples)
        return loss

    def policy(self, state):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor_model(s_tensor)
        return action.cpu().numpy().flatten()

    def save(self, file_path='./saved_models/', episode=None):
        torch.save(self.actor_model.state_dict(), file_path + "actor_model_" + str(episode) + ".pth")
        torch.save(self.critic_model.state_dict(), file_path + "critic_model_" + str(episode) + ".pth")

    def load(self, file_path='./saved_models/', episode=None):
        self.actor_model.load_state_dict(torch.load(file_path + 'actor_model_' + str(episode) + '.pth'))
        self.critic_model.load_state_dict(torch.load(file_path + 'critic_model_' + str(episode) + '.pth'))
