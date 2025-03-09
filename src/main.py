import os
import gymnasium as gym
import numpy as np
import optuna
import optuna.visualization as vis
import logging

from agents.abstract_agent import AbstractAgent
from agents.agent_factory import AgentFactory
from util.metrics_tracker import MetricsTracker
from util.dynamics import pendulum_dynamics
from util.compare_doa import compare_doa

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
# Count how many training log files already exist
existing_logs = [f for f in os.listdir(log_dir) if f.startswith("training_") and f.endswith(".log")]
log_number = len(existing_logs) + 1
log_filename = os.path.join(log_dir, f"training_{log_number}.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_filename)
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def show_last_episode(env_str: str, agent: AbstractAgent):
    env = gym.make(env_str, render_mode="human")
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
                if actor_loss:
                    ep_actor_losses.append(actor_loss)
                ep_critic_losses.append(critic_loss)

        cos_theta, sin_theta, _ = old_obs
        theta = np.arctan2(sin_theta, cos_theta)
        if abs(theta) < 0.3 * np.pi:
            total_stab += 1

        episode_returns.append(ep_return)
        avg_actor_loss = np.mean(ep_actor_losses) if ep_actor_losses else 0.0
        avg_critic_loss = np.mean(ep_critic_losses) if ep_critic_losses else 0.0
        episode_actor_losses.append(avg_actor_loss)
        episode_critic_losses.append(avg_critic_loss)

        if (episode + 1) % 1 == 0:
            logger.info(f"Episode {episode+1}/{num_episodes} | Return: {ep_return:.2f} "
                        f"| Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}")

    logger.info(f"Total stabilized: {total_stab}")

    if config.get("save_models"):
        agent.save()

    if config.get("show_last_episode", False):
        show_last_episode(env_str, agent)

    env.close()
    return episode_returns, episode_actor_losses, episode_critic_losses


def train_agent(env_str: str, config: dict, tracker: MetricsTracker, num_runs: int, num_episodes: int):
    if config["agent_str"] == "LQR":
        agent_id = "LQR"
    else:
        actor_arch = "-".join(map(str, config.get("actor_hidden_sizes", (64, 64))))
        critic_arch = "-".join(map(str, config.get("critic_hidden_sizes", (64, 64))))
        agent_id = f'AC_lr{config["actor_lr"]}_cr{config["critic_lr"]}_g{config["gamma"]}_n{config["n_steps"]}_a{actor_arch}_c{critic_arch}'
    
    logger.info(f"Training agent: {agent_id}")

    for run in range(num_runs):
        logger.info(f"Starting run {run+1}/{num_runs}...")
        returns, actor_losses, critic_losses = run_episode(env_str, config, num_episodes)
        tracker.add_run_returns(agent_id, returns)
        tracker.add_run_losses(agent_id, actor_losses, critic_losses)


def run_hyperparameter_optimization(env_str: str, 
                                    tracker: MetricsTracker, 
                                    num_episodes: int, 
                                    n_trials: int) -> optuna.study.Study:
    """
    Run hyperparameter optimization using Optuna by maximizing
    the agent's average return in the last 200 episodes of training.
    """

    def objective(trial: optuna.Trial) -> float:
        actor_lr = trial.suggest_float("actor_lr", 1e-5, 1e-2, log=True)
        critic_lr = trial.suggest_float("critic_lr", 1e-4, 1e-1, log=True)
        gamma = trial.suggest_float("gamma", 0.8, 0.9999)
        n_steps = trial.suggest_int("n_steps", 1, 1)
        actor_update_interval = trial.suggest_int("actor_update_interval", 2, 2)

        n_actor_layers = trial.suggest_int("n_actor_layers", 1, 3)
        actor_hidden_sizes = []
        for i in range(n_actor_layers):
            hidden_size = trial.suggest_int(f"actor_hidden_size_{i}", 8, 256)
            actor_hidden_sizes.append(hidden_size)

        n_critic_layers = trial.suggest_int("n_critic_layers", 1, 3)
        critic_hidden_sizes = []
        for i in range(n_critic_layers):
            hidden_size = trial.suggest_int(f"critic_hidden_size_{i}", 8, 256)
            critic_hidden_sizes.append(hidden_size)

        config = {
            "agent_str": "AC",
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "gamma": gamma,
            "n_steps": n_steps,
            "actor_update_interval": actor_update_interval,
            "actor_hidden_sizes": tuple(actor_hidden_sizes),
            "critic_hidden_sizes": tuple(critic_hidden_sizes),
            "save_models": False,
            "show_last_episode": False,
        }

        agent_id = (
            f"{config['agent_str']}_"
            f"lr{actor_lr:.1e}_cr{critic_lr:.1e}_g{gamma:.3f}_n{n_steps}_upd{actor_update_interval}_"
            f"a{'-'.join(map(str, actor_hidden_sizes))}_"
            f"c{'-'.join(map(str, critic_hidden_sizes))}"
        )

        print(f'Training {agent_id}')

        returns, actor_losses, critic_losses = run_episode(
            env_str=env_str,
            config=config,
            num_episodes=num_episodes
        )

        tracker.add_run_returns(agent_id, returns)
        tracker.add_run_losses(agent_id, actor_losses, critic_losses)

        if len(returns) >= 200:
            performance = np.mean(returns[-200:])
        else:
            print('Why are you here?')
            performance = np.mean(returns)

        trial.report(performance, step=num_episodes)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return performance

    # Create a study with direction = "maximize"
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study



def train_lyapunov_agent(env_str: str, config: dict, tracker: MetricsTracker, num_runs: int, num_episodes: int):
    actor_arch = "-".join(map(str, config.get("actor_hidden_sizes", (64, 64))))
    critic_arch = "-".join(map(str, config.get("critic_hidden_sizes", (64, 64))))
    agent_id = f'Ly_lr{config["actor_lr"]}_cr{config["critic_lr"]}_a{config["alpha"]}_n{config["n_steps"]}_a{actor_arch}_c{critic_arch}'


def train_ac_bayes_opt():
    env_str = "Pendulum-v1"
    num_episodes = 1000
    n_trials = 50

    tracker = MetricsTracker()

    study = run_hyperparameter_optimization(
        env_str, 
        tracker, 
        num_episodes, 
        n_trials
    )

    os.makedirs("plots", exist_ok=True)
    fig_history = optuna.visualization.plot_optimization_history(study)
    fig_history.write_image(os.path.join("plots", "optimization_history.png"))

    fig_importance = optuna.visualization.plot_param_importances(study)
    fig_importance.write_image(os.path.join("plots", "param_importances.png"))

    tracker.save_top10_plots(folder="plots")

    best_trial = study.best_trial
    logger.info("Hyperparameter optimization completed.")
    logger.info(f"Best trial params: {best_trial.params}")
    logger.info(f"Best trial value (avg return of last 200 episodes): {best_trial.value}")


def train_default():
    env_str = "Pendulum-v1"
    config_ac = {
        "agent_str": "AC",
        "actor_lr": 0.005,
        "critic_lr": 0.009,
        "gamma": 0.95,
        "n_steps": 1,
        "actor_update_interval": 2,
        "actor_hidden_sizes": (256, 64),
        "critic_hidden_sizes": (64, 64),
        "save_models": False,
        "show_last_episode": True,
    }

    num_runs = 1
    num_episodes = 500

    tracker = MetricsTracker()

    train_agent(env_str, config_ac, tracker, num_runs, num_episodes)

    tracker.save_top10_plots(folder="plots")

    logger.info("Hyperparameter optimization completed.")


def train_lac():
    env_str = "Pendulum-v1"
    config_lac = {
        "agent_str": "Lyapunov-AC",
        "alpha": 0.2,
        "actor_lr": 1e-3,
        "critic_lr": 2e-3,
        "dynamics_fn": pendulum_dynamics,
        "batch_size": 64,
        "num_paths_sampled": 8,
        "dt": 0.003,
        "norm_threshold": 5e-2,
        "integ_threshold": 150,
        "r1_bounds": (np.array([-1.0, -1.0, -8.0]), np.array([1.0, 1.0, 8.0])),
        "actor_hidden_sizes": (128, 64),
        "critic_hidden_sizes": (64, 64)
    }
        
    
    num_episodes = 1000

    tracker = MetricsTracker()
    
    agent = AgentFactory.create_agent(config=config_lac)

    ep_actor_losses = []
    ep_critic_losses = []

    for episode in range(num_episodes):
        loss = agent.update()

        if loss:
            actor_loss, critic_loss = loss
            ep_actor_losses.append(actor_loss)
            ep_critic_losses.append(critic_loss)

            print(f'Episode {episode + 1}/{num_episodes} - Actor Loss: {actor_loss}, Critic Loss: {critic_loss}')

    tracker.add_run_losses('lyAC', ep_actor_losses, ep_critic_losses)

    tracker.plot_split()

if __name__ == "__main__":
    train_ac_bayes_opt()
