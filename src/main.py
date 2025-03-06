import gymnasium as gym
import numpy as np
import optuna
import optuna.visualization as vis

from agents.abstract_agent import AbstractAgent
from agents.agent_factory import AgentFactory
from util.metrics_tracker import MetricsTracker
from util.compare_doa import compare_doa


def show_last_episode(env_str: str, agent: AbstractAgent):
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
    total_stab = 0

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

        cos_theta, sin_theta, theta_dot = old_obs
        theta = np.arctan2(sin_theta, cos_theta)

        if abs(theta) < 0.3 * np.pi:
            total_stab += 1

        episode_returns.append(ep_return)
        avg_actor_loss = np.mean(ep_actor_losses) if ep_actor_losses else 0.0
        avg_critic_loss = np.mean(ep_critic_losses) if ep_critic_losses else 0.0
        episode_actor_losses.append(avg_actor_loss)
        episode_critic_losses.append(avg_critic_loss)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} | Return: {ep_return:.2f} "
                  f"| Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}")

    print('Total stabilized: ', total_stab)
    if config.get("save_models"):
        agent.save()

    if config.get("show_last_episode", False):
        show_last_episode(env_str, agent)

    env.close()
    return episode_returns, episode_actor_losses, episode_critic_losses


def train_agent(env_str: str, config: dict, tracker: MetricsTracker, num_runs: int, num_episodes: int):
    # Construct agent name for AC agents using all relevant hyperparameters.
    if config["agent_str"] == "LQR":
        agent_id = "LQR"
    else:
        actor_arch = "-".join(map(str, config.get("actor_hidden_sizes", (64,64))))
        critic_arch = "-".join(map(str, config.get("critic_hidden_sizes", (64,64))))
        agent_id = f'AC_lr{config["actor_lr"]}_cr{config["critic_lr"]}_g{config["gamma"]}_n{config["n_steps"]}_a{actor_arch}_c{critic_arch}'
    
    print(f'Training agent: {agent_id}')

    for run in range(num_runs):
        print(f"Starting run {run+1}/{num_runs}...")
        returns, actor_losses, critic_losses = run_episode(env_str, config, num_episodes)
        tracker.add_run_returns(agent_id, returns)
        tracker.add_run_losses(agent_id, actor_losses, critic_losses)

def run_hyperparameter_optimization(env_str: str, tracker: MetricsTracker, num_episodes: int, n_trials: int):
    """
    Run hyperparameter optimization using Optuna.
    
    Parameters:
      config (dict): Base configuration dictionary (from main) that can contain default values.
      env_str (str): The environment string (e.g., "Pendulum-v1").
      num_episodes (int): The number of episodes to run per trial.
      n_trials (int): Number of Optuna trials.
      agent_prefix (str): A prefix for the agent name (default is "AC").
      
    Returns:
      study: The Optuna study object.
      tracker: The MetricsTracker with all recorded metrics.
    """
    
    def objective(trial):
        # Training Hyperparameter Suggestions
        actor_lr = trial.suggest_loguniform("actor_lr", 1e-4, 1e-2)
        critic_lr = trial.suggest_loguniform("critic_lr", 1e-3, 1e-1)
        gamma = trial.suggest_float("gamma", 0.8, 0.99)
        n_steps = trial.suggest_int("n_steps", 1, 10)

        # Architecture Tuning for the Actor
        n_actor_layers = trial.suggest_int("n_actor_layers", 1, 3)
        actor_hidden_sizes = []
        for i in range(n_actor_layers):
            hidden_size = trial.suggest_int(f"actor_hidden_size_{i}", 32, 128)
            actor_hidden_sizes.append(hidden_size)

        # Architecture Tuning for the Critic
        n_critic_layers = trial.suggest_int("n_critic_layers", 1, 3)
        critic_hidden_sizes = []
        for i in range(n_critic_layers):
            hidden_size = trial.suggest_int(f"critic_hidden_size_{i}", 32, 128)
            critic_hidden_sizes.append(hidden_size)

        # Build the Configuration Dictionary
        config = {
            "agent_str": "AC",
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "gamma": gamma,
            "n_steps": n_steps,
            "actor_hidden_sizes": tuple(actor_hidden_sizes),
            "critic_hidden_sizes": tuple(critic_hidden_sizes),
            "save_models": False,
            "show_last_episode": False,
        }

        agent_id = (
            f"{config['agent_str']}_lr{actor_lr:.1e}_cr{critic_lr:.1e}_g{gamma:.2f}_n{n_steps}_"
            f"a{'-'.join(map(str, actor_hidden_sizes))}_"
            f"c{'-'.join(map(str, critic_hidden_sizes))}"
        )

        print(f'Training agent: {agent_id}')

        # Run a Single Training Run
        returns, actor_losses, critic_losses = run_episode(env_str, config, num_episodes)

        # Record the Metrics
        tracker.add_run_returns(agent_id, returns)
        tracker.add_run_losses(agent_id, actor_losses, critic_losses)

        # Use the final episode's return as the performance measure.
        performance = returns[-1]
        trial.report(performance, step=num_episodes)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return performance

    # Create and run the Optuna study.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    return study
 

def main():
    env_str = "Pendulum-v1"
    config_ac = {
        "agent_str": "AC",
        "actor_lr": 0.0005,
        "critic_lr": 0.009,
        "gamma": 0.9,
        "n_steps": 5,
        "save_models": False,
        "show_last_episode": True,
    }

    config_lqr = {
        "agent_str": "LQR",
        "g": 10.0,
        "Q": np.diag([0.001, 0.001]),
        "R": np.array([[0.0001]]),
        "save_models": True,
        "show_last_episode": False,
    }
    num_runs = 1
    num_episodes = 500

    n_trials = 200

    tracker = MetricsTracker()

    #train_agent(env_str, config_ac, tracker, num_runs, num_episodes)
    
    study = run_hyperparameter_optimization(env_str, tracker, num_episodes, n_trials)

    vis.plot_optimization_history(study).show()
    vis.plot_param_importances(study).show()

    tracker.plot_split()

if __name__ == "__main__":
    main()
