import gymnasium as gym
import numpy as np

from agents.abstract_agent import AbstractAgent
from agents.agent_factory import AgentFactory
from util.metrics_tracker import MetricsTracker
from util.compare_doa import compare_doa


def show_one_episode(env_str: str, agent: AbstractAgent):
    env = gym.make(env_str, render_mode="human", )

    done = False
    obs, _ = env.reset() 
    while not done:
        action = agent.policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()


def run_episode(env_str: str, config: dict, num_episodes: int):
    env = gym.make(env_str)
    agent = AgentFactory.create_agent(config=config, env=env)

    episode_returns = []
    episode_actor_losses = []
    episode_critic_losses = []

    for episode in range(num_episodes):
        ep_return = 0.0
        ep_actor_losses = []
        ep_critic_losses = []
        done = False
        obs, _ = env.reset() 

        while not done:
            old_obs = obs
            action = agent.policy(old_obs)
            obs, reward, terminated, truncated, _ = env.step(action)

            ep_return += reward
            done = terminated or truncated

            agent.add_transition((old_obs, action, reward, obs, done))

        if len(agent._replay_buffer) > 0:
            loss = agent.update()
            if loss:
                actor_loss, critic_loss = loss
                ep_actor_losses.append(actor_loss)
                ep_critic_losses.append(critic_loss)

        episode_returns.append(ep_return)
        avg_actor_loss = np.mean(ep_actor_losses) if ep_actor_losses else 0.0
        avg_critic_loss = np.mean(ep_critic_losses) if ep_critic_losses else 0.0
        episode_actor_losses.append(avg_actor_loss)
        episode_critic_losses.append(avg_critic_loss)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Return: {ep_return:.2f} "
                  f"| Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}")

    if config.get("save_models"):
        agent.save()

    if config.get("show_one_episode", False):
        show_one_episode(env_str, agent)

    env.close()
    return episode_returns, episode_actor_losses, episode_critic_losses


def train_agent(env_str: str, config: dict, tracker: MetricsTracker, num_runs: int, num_episodes: int):
    if config["agent_str"] == "LQR":
        agent_name = "LQR"
    else:
        agent_name = f'{config["agent_str"]}_lr{config["actor_lr"]}_{config["critic_lr"]}_gamma{config["gamma"]}_n{config["n_steps"]}'
    
    print(agent_name)

    for run in range(num_runs):
        print(f"Starting run {run+1}/{num_runs}...")
        returns, actor_losses, critic_losses = run_episode(env_str, config, num_episodes)
        tracker.add_run_returns(agent_id=agent_name, returns=returns)
        tracker.add_run_losses(agent_id=agent_name, actor_losses=actor_losses, critic_losses=critic_losses)
 
def main():
    env_str = "Pendulum-v1"
    config_ac = {
        "agent_str": "ACTOR-CRITIC",
        "actor_lr": 0.0005,
        "critic_lr": 0.0009,
        "gamma": 0.9,
        "n_steps": 5,
        "save_models": True,
    }
    config_lqr = {
        "agent_str": "LQR",
        "g": 10.0,
        "save_models": True,
        "show_one_episode": False,
    }
    num_runs = 1
    num_episodes = 1000

    tracker = MetricsTracker()

    train_agent(env_str, config_ac, tracker, num_runs, num_episodes)
    train_agent(env_str, config_lqr, tracker, num_runs, num_episodes)
    
    tracker.plot_split()

    compare_doa(config_ac, config_lqr)

if __name__ == "__main__":
    main()
