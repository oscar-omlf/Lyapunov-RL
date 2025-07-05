import os
import logging
import json
import numpy as np
import torch
from copy import deepcopy

from util.rk4_step import rk4_step
from agents.agent_factory import AgentFactory
from agents.lqr_agent import LQRAgent
from agents.abstract_agent import AbstractAgent
from util.metrics_tracker import MetricsTracker
from util.logger_utils import setup_run_directory_and_logging
from util.csv_utils import _write_2d_csv, _write_counts_csv
from util.dynamics import (
    pendulum_dynamics_np,
    compute_pendulum_reward
)
from config import (
    config_ldp_pendulum, 
    config_lac_pendulum, 
    config_las_td3_pendulum,
    config_td3_pendulum, 
    config_lqr_pendulum
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DT = 0.003
NUM_RUNS = 15
NUM_EPISODES = 500
NUM_STEPS_PER_EPISODE = 3000
STABILIZATION_THRESHOLD = 0.0005
CONSECUTIVE_STABLE_THRESHOLD = 10

# 0.1 to 0.9 with step of 0.1, using range() to get floats
BETAS = [round(x * 0.1, 2) for x in range(1, 4)]
CFG_LST = []

for beta in BETAS:
    cfg = deepcopy(config_las_td3_pendulum)
    cfg["beta"] = beta
    CFG_LST.append(cfg)


dynamics_fn = pendulum_dynamics_np
rewards_fn = compute_pendulum_reward

CFG_EVAL = {
    "model_name": "EVALUATION",
    "environment": "InvertedPendulum",
}

def run_model_evaluation(
        agent: AbstractAgent, 
        model_name: str,
        run_dir: str, 
        logger: logging.Logger,
        tracker: MetricsTracker
    ):
    # Containers for raw data
    returns_all = np.empty((NUM_RUNS, NUM_EPISODES), dtype=np.float32)
    steps_all = np.full((NUM_RUNS, NUM_EPISODES), np.nan, dtype=np.float32)
    counts_all = np.empty(NUM_RUNS, dtype=np.int32)

    for run in range(NUM_RUNS):
        logger.info(f"[{model_name}] Starting run {run+1}/{NUM_RUNS}")

        run_stabilization_count = 0

        for episode in range(NUM_EPISODES):
            current_state = np.array([
                np.random.uniform(-np.pi, np.pi),   
                np.random.uniform(-8.0, 8.0)
            ])
        
            episode_reward = 0.0
            consecutive_stable = 0
            episode_stabilized = False
            steps_to_stabilize = np.nan

            for step in range(NUM_STEPS_PER_EPISODE):
                if isinstance(agent, LQRAgent):
                    action = agent.policy_np(current_state)
                else:
                    with torch.no_grad():
                        action = agent.policy(current_state)

                next_state = rk4_step(dynamics_fn, current_state, action, DT).squeeze()

                # wrap the angle to [-pi, pi] and clip the velocity to [-8, 8]
                next_state[0] = np.mod(next_state[0] + np.pi, 2.0 * np.pi) - np.pi
                next_state[1] = np.clip(next_state[1], -8.0, 8.0)

                reward_float = rewards_fn(current_state, float(action))

                episode_reward += reward_float
                current_state = next_state

                # Check for stability
                if np.linalg.norm(current_state, ord=2) < STABILIZATION_THRESHOLD:
                    consecutive_stable += 1
                    if (
                        consecutive_stable >= CONSECUTIVE_STABLE_THRESHOLD
                        and episode_stabilized is False
                    ):
                        episode_stabilized = True
                        steps_to_stabilize = step + 1
                        run_stabilization_count += 1
                else:
                    consecutive_stable = 0
               
            returns_all[run, episode] = episode_reward
            steps_all[run, episode] = steps_to_stabilize

        counts_all[run] = run_stabilization_count
        tracker.add_run_returns(model_name, returns_all[run].tolist())
        logger.info(
            f"[{model_name}] Run {run+1} complete."
            f"  Average return: {np.mean(returns_all[run]):.2f} +/- {np.std(returns_all[run]):.2f}"
            f"  Episodes with stabilization: {run_stabilization_count} / {NUM_EPISODES} episodes"
            f"  Steps to stabilize: {np.nanmean(steps_all[run]):.2f} +/- {np.nanstd(steps_all[run]):.2f}"
        )

    agent_dir = os.path.join(run_dir, model_name)
    os.makedirs(agent_dir, exist_ok=True)
    _write_2d_csv(returns_all, os.path.join(agent_dir, "returns.csv"))
    _write_2d_csv(steps_all, os.path.join(agent_dir, "steps_to_stabilize.csv"))
    _write_counts_csv(counts_all, os.path.join(agent_dir, "episodes_stabilized.csv"))

    mean_returns = np.mean(returns_all)
    std_returns = np.std(returns_all)

    mean_steps = np.nanmean(steps_all)
    std_steps = np.nanstd(steps_all)

    mean_stab = np.mean(counts_all)
    std_stab = np.std(counts_all)

    logger.info(f"Aggregated Statistics (mean +- SD across runs):")
    logger.info(f"  Mean Return: {mean_returns:.2f} +/- {std_returns:.2f}")
    logger.info(f"  Mean Steps to Stabilize: {mean_steps:.2f} +/- {std_steps:.2f}")
    logger.info(f"  Mean Episodes with Stabilization: {mean_stab:.2f} +/- {std_stab:.2f}")


def main():
    tracker = MetricsTracker()
    run_dir, logger = setup_run_directory_and_logging(CFG_EVAL, evaluation=True)
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Raw CSVs will be saved to: {run_dir}")
    logger.info(f"Using stabilization threshold: {STABILIZATION_THRESHOLD}")

    for cfg in CFG_LST:
        cfg_str = json.dumps(cfg, indent=4, default=str)
        logger.info(cfg_str)
        model_name = cfg["model_name"]
        beta = cfg.get("beta", None) 
        agent = AgentFactory.create_agent(config=cfg)

        if model_name != "LQR":
            if beta is not None:
                agent.load(f'./best_models/{model_name}/{cfg["beta"]}/', episode=0)
                model_name = f"{model_name}_{beta}"
            else:
                agent.load(f'./best_models/{model_name}/', episode=0)

        run_model_evaluation(agent, model_name, run_dir, logger, tracker)
    
    tracker.save_top10_plots(run_dir)

if __name__ == "__main__":
    main()
