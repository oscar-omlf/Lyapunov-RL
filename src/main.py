import gymnasium as gym
import numpy as np
import torch

from agents.agent_factory import AgentFactory
from util.metrics_tracker import MetricsTracker

def env_interaction(env_str: str, config: dict, num_episodes: int) -> tuple[list, list, list]:
    """
    Interact with the environment for a given number of episodes.
    For each episode, the agent collects transitions, updates its weights,
    and we record:
      - the cumulative return for that episode,
      - the average actor loss over that episode (if any update occurred),
      - the average critic loss over that episode.
    """
    env = gym.make(env_str)
    obs, _ = env.reset()

    agent = AgentFactory.create_agent(config=config, env=env)

    episode_returns = []
    episode_actor_losses = []
    episode_critic_losses = []

    for episode in range(num_episodes):
        ep_return = 0.0
        ep_actor_losses = []
        ep_critic_losses = []
        done = False

        while not done:
            old_obs = obs
            action = agent.policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward

            agent.add_transition((old_obs, action, reward, obs))
            loss = agent.update()

            if loss:
                actor_loss, critic_loss = loss
                ep_actor_losses.append(actor_loss)
                ep_critic_losses.append(critic_loss)
            done = terminated or truncated

        episode_returns.append(ep_return)
        avg_actor_loss = np.mean(ep_actor_losses) if ep_actor_losses else 0.0
        avg_critic_loss = np.mean(ep_critic_losses) if ep_critic_losses else 0.0
        episode_actor_losses.append(avg_actor_loss)
        episode_critic_losses.append(avg_critic_loss)

        obs, _ = env.reset()
        # print(f"Episode {episode+1}: Return={ep_return:.2f}")

    env.close()
    return episode_returns, episode_actor_losses, episode_critic_losses

def execute_agent_runs(env_str: str, agent_str: str, num_runs: int, num_episodes: int,
                       gamma: float, actor_lr: float, critic_lr: float) -> tuple[list, list, list]:
    """
    Run the specified agent configuration for num_runs runs.
    Returns three lists:
      - returns_list: list of lists (each run's episode returns),
      - actor_loss_list: list of lists (each run's per-episode actor losses),
      - critic_loss_list: list of lists (each run's per-episode critic losses).
    """
    returns_list = []
    actor_loss_list = []
    critic_loss_list = []

    for run in range(num_runs):
        print(f"\nRun {run+1} of {num_runs}")

        config = {
            "agent_str": agent_str,
            "n_steps": 3,
            "gamma": gamma,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

        ret, a_loss, c_loss = env_interaction(env_str, config, num_episodes)
        returns_list.append(ret)
        actor_loss_list.append(a_loss)
        critic_loss_list.append(c_loss)

    return returns_list, actor_loss_list, critic_loss_list

def run_set_parameters(env_str: str, agent_str: str, tracker: MetricsTracker,
                       num_runs: int, num_episodes: int):
    """
    Run the default configuration for the given agent type on the environment.
    The run's metrics are added to the tracker.
    """
    gamma, actor_lr, critic_lr = 0.99, 0.0001, 0.0005
    print(f"Default hyperparameters: Gamma={gamma}, Actor LR={actor_lr}, Critic LR={critic_lr}")

    returns_list, actor_losses, critic_losses = execute_agent_runs(
        env_str, agent_str, num_runs, num_episodes, gamma, actor_lr, critic_lr
    )

    for run_returns in returns_list:
        tracker.add_run_returns(agent_str, run_returns)
    for run_actor_losses, run_critic_losses in zip(actor_losses, critic_losses):
        tracker.add_run_losses(agent_str, run_actor_losses, run_critic_losses)

    episodes, ret_means, ret_ses = tracker.get_avg_returns(agent_str)
    if episodes:
        print(f"Final Episode {episodes[-1]} - Average Return: {ret_means[-1]:.2f} (SE: {ret_ses[-1]:.2f})")

    tracker.plot_split()

def main():
    env_str = "Pendulum-v1"
    agent_str = "RANDOM"
    tracker = MetricsTracker()

    num_runs = 10
    num_episodes = 200

    run_set_parameters(env_str, agent_str, tracker, num_runs, num_episodes)

if __name__ == "__main__":
    main()
