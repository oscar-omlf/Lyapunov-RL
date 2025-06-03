import os
import gymnasium as gym
import numpy as np
import optuna
import logging

from agents.abstract_agent import AbstractAgent
from agents.agent_factory import AgentFactory
from util.metrics_tracker import MetricsTracker
from util.dynamics import (
    pendulum_dynamics_torch,
    pendulum_dynamics_dreal,
    double_integrator_dynamics_torch,
    vanderpol_dynamics_torch
)

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


def update_agent(agent: AbstractAgent, ep_actor_losses: list, ep_critic_losses: list):
    loss = agent.update()
    if loss:
        actor_loss, critic_loss = loss
        if actor_loss:
            ep_actor_losses.append(actor_loss)
        ep_critic_losses.append(critic_loss)


def run_episode(env_str: str, config: dict, num_episodes: int):
    env = gym.make(env_str)
    agent = AgentFactory.create_agent(config=config, env=env)

    episode_returns = []
    episode_actor_losses = []
    episode_critic_losses = []
    total_stab = 0

    start_episodes = config.get("start_episodes", 125)
    update_threshold = config.get("batch_size", 256)

    agent_str = config.get("agent_str")

    for episode in range(num_episodes):
        ep_return = 0.0
        ep_actor_losses = []
        ep_critic_losses = []
        done = False
        obs, _ = env.reset()

        while not done:
            old_obs = obs
            if agent_str == "TD3":
                # Use random actions during the initial exploration phase.
                if episode < start_episodes:
                    action = env.action_space.sample()
                else:
                    action = agent.policy(old_obs)
            else:
                action = agent.policy(old_obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
            agent.add_transition((old_obs, action, reward, obs, done))

            # Only update if we're past the random phase.
            if episode >= start_episodes and (len(agent._replay_buffer) >= update_threshold or done):
                update_agent(agent, ep_actor_losses, ep_critic_losses)

        # Calculate stability (assumes pendulum state: cos, sin, theta_dot)
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

        agent_id = f'{config["agent_str"]}_lr{config["actor_lr"]}_cr{config["critic_lr"]}_g{config["gamma"]}'
    
        if config["agent_str"] == "AC":
            agent_id += f'n{config["n_steps"]}'
        
        agent_id += f'_a{actor_arch}_c{critic_arch}'
    
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
        policy_freq = trial.suggest_int("policy_freq", 1, 5)

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
            "policy_freq": policy_freq,
            "actor_hidden_sizes": tuple(actor_hidden_sizes),
            "critic_hidden_sizes": tuple(critic_hidden_sizes),
            "save_models": False,
            "show_last_episode": False,
        }

        agent_id = (
            f"{config['agent_str']}_"
            f"lr{actor_lr:.1e}_cr{critic_lr:.1e}_g{gamma:.3f}_n{n_steps}_upd{policy_freq}_"
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
    config_lqr = {
        "agent_str": "LQR",
        "g": 9.81,
        "save_models": True,
        "show_last_episode": False,
    }

    config_td3 = {
        "agent_str": "TD3",
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 256,
        "policy_freq": 2,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "start_episodes": 125,
        "expl_noise": 0.1,
        "actor_hidden_sizes": (256, 256),
        "critic_hidden_sizes": (256, 256),
        "save_models": False,
        "show_last_episode": False,
    }

    num_runs = 1
    num_episodes = 500

    tracker = MetricsTracker()

    train_agent(env_str, config_td3, tracker, num_runs, num_episodes)

    tracker.save_top10_plots(folder="plots")

    logger.info("Training completed.")


def train_lac():
    config_lac = {
        "agent_str": "Lyapunov-AC",
        "alpha": 0.1,
        "actor_lr": 3e-3,
        "critic_lr": 2e-3,
        "dynamics_fn": pendulum_dynamics_torch,
        "dynamics_fn_dreal": pendulum_dynamics_dreal,
        "batch_size": 64,
        "num_paths_sampled": 8,
        "dt": 0.01,
        "norm_threshold": 5e-2,
        "integ_threshold": 50,
        "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])), 
        "actor_hidden_sizes": (30, 30),
        "critic_hidden_sizes": (30, 30),
        "state_space": np.zeros(2),
        "action_space":np.zeros(1),
        "max_action": 1.0
    }

    config_lqr = {
        "agent_str": "LQR",
        "g": 9.81,
        "l": 0.5,
        "m": 0.15,
        "state_space": np.zeros(2),
        "action_space": np.zeros(1),
        "max_action": 1.0,
        "x_star": np.array([0.0, 0.0])
    }

    config_las_lac = {
        "agent_str": "LAS-LAC",
        "LQR": config_lqr,
        "LAC": config_lac,
        "beta": 0.5,
        "dynamics_fn": pendulum_dynamics_torch,
        "dynamics_fn_dreal": pendulum_dynamics_dreal,
        "state_space": np.zeros(2),
        "action_space": np.zeros(1),
        "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0]))
    }
    
    num_episodes = 3000

    tracker = MetricsTracker()
    
    agent = AgentFactory.create_agent(config=config_las_lac)

    ep_actor_losses = []
    ep_critic_losses = []

    for episode in range(num_episodes):
        loss = agent.update()

        if loss:
            actor_loss, critic_loss = loss
            ep_actor_losses.append(actor_loss)
            ep_critic_losses.append(critic_loss)

        logger.info(f"Episode {episode+1}/{num_episodes} | Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f}")

        if (episode + 1) % 1000 == 0:
            agent.save()

    tracker.add_run_losses('lyAC', ep_actor_losses, ep_critic_losses)

    tracker.save_top10_losses_plot(folder='plots')

if __name__ == "__main__":
    train_default()
