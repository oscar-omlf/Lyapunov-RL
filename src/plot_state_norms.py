#!/usr/bin/env python3
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from util.rk4_step import rk4_step
from agents.agent_factory import AgentFactory
from agents.lqr_agent import LQRAgent
from util.dynamics import pendulum_dynamics_np
from config import (
    config_lac_pendulum,
    config_td3_pendulum,
    config_lqr_pendulum,
    config_ldp_pendulum,
    config_las_td3_pendulum
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = 0.003
NUM_EPISODES = 500
NUM_STEPS = 3000
STABILIZATION_THRESHOLD = 0.0005

AGENTS = {
    "LAC":         (config_lac_pendulum,    "./best_models/LAC/"),
    "TD3":         (config_td3_pendulum,    "./best_models/TD3/"),
    "LQR":         (config_lqr_pendulum,    "./best_models/LQR/"),
    "LDP_0.9":     (dict(config_ldp_pendulum, **{"beta":0.9}),    "./best_models/LDP/0.9/"),
    "LAS_TD3_0.9": (dict(config_las_td3_pendulum, **{"beta":0.9}),"./best_models/LAS_TD3/0.9/")
}

dynamics_fn = pendulum_dynamics_np

def simulate_controller(agent, config):
    norms = np.zeros((NUM_EPISODES, NUM_STEPS), dtype=np.float32)
    thetas = np.zeros_like(norms)
    theta_dots = np.zeros_like(norms)

    for ep in range(NUM_EPISODES):
        state = np.array([
            np.random.uniform(-2 * np.pi, 0),
            np.random.uniform(-8.0, 8.0)
        ], dtype=np.float32)

        for t in range(NUM_STEPS):            
            if isinstance(agent, LQRAgent):
                action = agent.policy_np(state)
            else:
                with torch.no_grad():
                    action = agent.policy(state)

            next_state = rk4_step(dynamics_fn, state, action, DT).squeeze()

            norms[ep, t] = np.linalg.norm(next_state, ord=2)

            thetas[ep, t] = next_state[0]

            theta_dots[ep, t] = next_state[1]

            state = next_state

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{NUM_EPISODES} complete.")

    return norms, thetas, theta_dots

def main():
    mean_curves = {"norm":{}, "theta":{}, "theta_dot":{}}
    se_curves   = {"norm":{}, "theta":{}, "theta_dot":{}}

    os.makedirs("plots", exist_ok=True)
    os.makedirs("plots/5", exist_ok=True)

    for name, (cfg, load_path) in AGENTS.items():
        print(f"\n=== Simulating {name} ===")
        agent = AgentFactory.create_agent(config=cfg)
        if name != "LQR":
            agent.load(load_path, episode=0)

        norms, thetas, theta_dots = simulate_controller(agent, cfg)

        for key, arr in [("norm", norms), ("theta", thetas), ("theta_dot", theta_dots)]:
            mean = arr.mean(axis=0)
            se   = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            mean_curves[key][name] = mean
            se_curves[key][name]   = se

            plt.figure(figsize=(8,4))
            x = np.arange(NUM_STEPS)
            plt.plot(x, mean, label=name)
            plt.fill_between(x, mean-se, mean+se, alpha=0.2)
            plt.axhline(
                STABILIZATION_THRESHOLD if key=="norm" else 0,
                linestyle="--", color="k",
                label=("Threshold" if key=="norm" else "0")
            )
            plt.title(f"{key.replace('_',' ').title()} over Steps - {name}")
            plt.xlabel("Step")
            plt.ylabel({"norm":"||state||₂","theta":"θ","theta_dot":"θ̇"}[key])
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"./plots/5/{name}_{key}.png")
            plt.close()

    x = np.arange(NUM_STEPS)
    for key in ["norm","theta","theta_dot"]:
        plt.figure(figsize=(10,6))
        for name in AGENTS:
            mean = mean_curves[key][name]
            se   = se_curves[key][name]
            plt.plot(x, mean, label=name)
            plt.fill_between(x, mean-se, mean+se, alpha=0.1)
        plt.axhline(
            STABILIZATION_THRESHOLD if key=="norm" else 0,
            linestyle="--", color="k"
        )
        plt.title(f"{key.replace('_',' ').title()} over Steps")
        plt.xlabel("Step")
        plt.ylabel({"norm":"||state||₂","theta":"θ","theta_dot":"θ̇"}[key])
        plt.legend(loc="upper right", fontsize="small")
        plt.tight_layout()
        plt.savefig(f"./plots/5/all_{key}.png")
        plt.close()

    plt.figure(figsize=(10,6))
    for name in AGENTS:
        mean = mean_curves["theta"][name]
        se   = se_curves["theta"][name]
        plt.plot(x, mean, label=name)
        plt.fill_between(x, mean-se, mean+se, alpha=0.1)
    plt.axhline(0, linestyle="--", color="k")
    plt.title("θ over Steps")
    plt.xlabel("Step")
    plt.ylabel("θ")
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig("./plots/5/all_theta.png")
    plt.close()

    plt.figure(figsize=(10,6))
    for name in AGENTS:
        if name == "TD3":
            continue
        mean = mean_curves["theta"][name]
        se   = se_curves["theta"][name]
        plt.plot(x, mean, label=name)
        plt.fill_between(x, mean-se, mean+se, alpha=0.1)
    plt.axhline(0, linestyle="--", color="k")
    plt.title("θ over Steps")
    plt.xlabel("Step")
    plt.ylabel("θ")
    plt.legend(loc="upper right", fontsize="small")
    plt.tight_layout()
    plt.savefig("./plots/5/noTD3_theta.png")
    plt.close()

    print("\nPlots saved to ./plots/")

if __name__ == "__main__":
    main()
