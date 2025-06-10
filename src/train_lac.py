import os
import numpy as np

from agents.agent_factory import AgentFactory
from util.dynamics import pendulum_dynamics_torch, pendulum_dynamics_dreal
from util.metrics_tracker import MetricsTracker
from util.logger_utils import setup_run_directory_and_logging


def train_lac():
    config_lac = {
        "agent_str": "Lyapunov-AC",
        "model_name": "LAC",
        "alpha": 0.2,
        "lr": 2e-3,
        "dynamics_fn": pendulum_dynamics_torch,
        "dynamics_fn_dreal": pendulum_dynamics_dreal,
        "batch_size": 128,
        "num_paths_sampled": 8,
        "dt": 0.003,
        "norm_threshold": 5e-2,
        "integ_threshold": 150,
        "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
        "actor_hidden_sizes": (5, 5),
        "critic_hidden_sizes": (20, 20),
        "state_space": np.zeros(2),
        "action_space": np.zeros(1),
        "max_action": 1.0
    }
    
    num_episodes = 3000
    model_name = config_lac["model_name"]

    run_dir, logger = setup_run_directory_and_logging(config_lac)
    tracker = MetricsTracker()
    
    config_lac["run_dir"] = run_dir
    agent = AgentFactory.create_agent(config=config_lac)

    ep_actor_losses = []
    ep_critic_losses = []

    for episode in range(num_episodes):
        loss = agent.update()

        if loss:
            actor_loss, critic_loss = loss
            ep_actor_losses.append(actor_loss)
            ep_critic_losses.append(critic_loss)

        if (episode + 1) % 100 == 0:
             logger.info(f"Episode {episode+1}/{num_episodes} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")

        if (episode + 1) % 1000 == 0:
            agent.save(file_path=run_dir, episode=(episode + 1)) 
            logger.info(f"Model weights saved to {run_dir}")

    logger.info("Training Finished")

    tracker.add_run_losses(model_name, ep_actor_losses, ep_critic_losses)
    tracker.save_top10_losses_plot(folder=run_dir)
    logger.info(f"Loss plot saved to {run_dir}")

if __name__ == "__main__":
    train_lac()
