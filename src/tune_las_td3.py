from __future__ import annotations
import argparse
import copy
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

from util.rk4_step import rk4_step
from util.dynamics import (
    pendulum_dynamics_np,
    pendulum_dynamics_dreal,
    compute_pendulum_reward,
)
from agents.las_td3_agent import LAS_TD3Agent
from util.metrics_tracker import MetricsTracker
from util.logger_utils import setup_run_directory_and_logging
from config import config_las_td3_pendulum


def make_base_config() -> dict:
    DT = 0.003
    PENDULUM_G = 9.81
    PENDULUM_M = 0.15
    PENDULUM_L = 0.5
    PENDULUM_B = 0.1
    MAX_ACTION_VAL = 1.0

    config = config_las_td3_pendulum
    config

    config = {
        "model_name": "LAS_TD3_BetaSearch",
        "environment": "InvertedPendulum",
        "max_action": MAX_ACTION_VAL,
        "dynamics_fn_dreal": pendulum_dynamics_dreal,
        "LQR": {
            "agent_str": "LQR",
            "environment": "InvertedPendulum",
            "discrete_discounted": True,
            "gamma": 0.99,
            "dt": DT,
            "g": PENDULUM_G,
            "m": PENDULUM_M,
            "l": PENDULUM_L,
            "b": PENDULUM_B,
            "max_action": MAX_ACTION_VAL,
            "state_space": np.zeros(2),
            "action_space": np.zeros(1),
        },
        "gamma": 0.9,
        "tau": 0.005,
        "policy_freq": 2,
        "batch_size": 256,
        "policy_noise": 0.2,
        "noise_clip": 0.5,
        "expl_noise": 0.1,
        "actor_lr": 3e-4,
        "critic_lr": 3e-4,
        "actor_hidden_sizes": (256, 256),
        "critic_hidden_sizes": (256, 256),
        "state_space": np.zeros(2),
        "action_space": np.zeros(1),
        "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
        "c_star": 1.1982,
    }

    return config


def train_once(
    beta: float,
    num_episodes: int,
    steps_per_ep: int,
    base_config: dict,
    parent_run_dir: Path,
    logger,
    seed: int = 0,
):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[beta={beta:.2f} | seed={seed}] Using device: {DEVICE}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = copy.deepcopy(base_config)
    cfg["beta"] = beta
    cfg["model_name"] = f"{cfg['model_name']}_beta{beta:.2f}"

    sub_dir = parent_run_dir / f"beta_{beta:.2f}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    cfg["run_dir"] = str(sub_dir)

    agent = LAS_TD3Agent(cfg)

    initial_exploration_steps = 1000
    total_steps_taken = 0
    LOG_EVERY_STEP = 50

    returns: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []

    for ep in range(num_episodes):
        ep_actor_losses, ep_critic_losses = [], []
        state = np.array([
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-8.0, 8.0),
        ])
        ep_return = 0.0

        for step in range(steps_per_ep):
            if total_steps_taken < initial_exploration_steps:
                action = np.random.uniform(-cfg["max_action"], cfg["max_action"], size=(agent.action_dim,))
            else:
                action = agent.policy(state)

            next_state = rk4_step(pendulum_dynamics_np, state, action, 0.03).squeeze()
            next_state[0] = (next_state[0] + np.pi) % (2 * np.pi) - np.pi
            next_state[1] = np.clip(next_state[1], -8.0, 8.0)
            reward = compute_pendulum_reward(state, float(action))

            ep_return += reward
            total_steps_taken += 1
            done = step == steps_per_ep - 1

            agent.add_transition(
                (state, action, np.array([reward]), next_state, np.array([float(done)]))
            )

            if total_steps_taken > initial_exploration_steps:
                a_loss, c_loss = agent.update()
                if a_loss is not None:
                    ep_actor_losses.append(a_loss)
                if c_loss is not None:
                    ep_critic_losses.append(c_loss)

            if step % LOG_EVERY_STEP == 0:
                logger.info(
                    "[beta=%.2f seed=%d] Ep %d Step %d | Ret %.2f",
                    beta,
                    seed,
                    ep + 1,
                    step + 1,
                    ep_return,
                )

            state = next_state
            if done:
                break

        returns.append(ep_return)
        if ep_actor_losses:
            actor_losses.append(float(np.mean(ep_actor_losses)))
        if ep_critic_losses:
            critic_losses.append(float(np.mean(ep_critic_losses)))

    agent.save(str(sub_dir))
    return returns, actor_losses, critic_losses, agent



def main():
    parser = argparse.ArgumentParser("Grid-search beta for LAS-TD3 with multiple seeds")
    parser.add_argument("--betas", nargs="*", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--runs", type=int, default=5, help="Independent runs per beta (default 5)")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    args = parser.parse_args()

    base_cfg = make_base_config()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_cfg["model_name"] += f"_{ts}"

    run_dir, logger = setup_run_directory_and_logging(base_cfg)
    base_cfg["run_dir"] = str(run_dir)
    logger.info("beta grid search list: %s", args.betas)

    tracker = MetricsTracker()

    best_beta, best_score = None, -np.inf
    results: list[tuple[float, float]] = []

    for beta in args.betas:
        agent_id = f"LAS_TD3_beta{beta:.2f}"
        sub_dir = Path(run_dir) / f"beta_{beta:.2f}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        for run_idx in range(args.runs):
            returns, a_losses, c_losses, agent = train_once(
                beta,
                args.episodes,
                args.steps,
                base_cfg,
                Path(run_dir),
                logger,
                seed=args.seed + run_idx,
            )

            tracker.add_run_returns(agent_id, returns)
            tracker.add_run_losses(agent_id, a_losses, c_losses)

            episodes, means, _ = tracker.get_avg_returns(agent_id)
            current_score = float(np.mean(means))
            logger.info(
                "[beta=%.2f] Run %d/%d done | aggregated mean %.2f",
                beta,
                run_idx + 1,
                args.runs,
                current_score,
            )

        tracker.save_top10_plots(folder=str(sub_dir))

        episodes, means, _ = tracker.get_avg_returns(agent_id)
        beta_score = float(np.mean(means))
        results.append((beta, beta_score))
        if beta_score > best_score:
            best_beta, best_score = beta, beta_score
        logger.info("[beta=%.2f] aggregated score after %d runs: %.2f", beta, args.runs, beta_score)

    logger.info("beta grid search summary")
    for beta, score in results:
        logger.info("beta = %.2f | aggregated mean return = %.2f", beta, score)
    logger.info("Best beta = %.2f with aggregated mean return = %.2f", best_beta, best_score)


if __name__ == "__main__":
    main()
