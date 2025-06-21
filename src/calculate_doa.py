#!/usr/bin/env python3
import os
import numpy as np
import torch
import csv
from pathlib import Path

from util.doa_utils import estimate_doa
from agents.agent_factory import AgentFactory
from agents.lqr_agent import LQRAgent
from config import config_lac_pendulum, config_ldp_pendulum, config_lqr_pendulum

# ---------------------- SETTINGS ---------------------- #
# Lyapunov-based configs to evaluate
LYAP_CONFIGS = [config_lac_pendulum, config_ldp_pendulum, config_lqr_pendulum]
# User-defined DoA level per model
LEVELS = {
    "LAC": 0.9593,
    "LDP": 0.94,
    "LQR": 1.1522,
}
# Monte Carlo samples for AC agents
N_SAMPLES = 50000
# Output CSV path
OUTPUT_CSV = Path("doa_results.csv")
# ------------------------------------------------------ #


def compute_lqr_doa_analytic(P: np.ndarray, level: float, lb: np.ndarray, ub: np.ndarray) -> float:
    """
    Analytically compute the area fraction of the ellipsoid {x: x^T P x <= level}
    within the rectangle defined by lb and ub in 2D.
    """
    # Ellipse area = pi * level / sqrt(det(P))
    detP = np.linalg.det(P)
    if detP <= 0:
        return 0.0
    ellipse_area = np.pi * level / np.sqrt(detP)
    # Box area
    box_area = np.prod(ub - lb)
    # Fraction
    frac = ellipse_area / box_area
    return float(min(max(frac, 0.0), 1.0))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for cfg in LYAP_CONFIGS:
        model_name = cfg["model_name"]
        # bounds
        lb, ub = cfg["r1_bounds"]
        lb = np.array(lb, dtype=np.float32)
        ub = np.array(ub, dtype=np.float32)
        level = LEVELS.get(model_name, 0.0)

        # instantiate
        agent = AgentFactory.create_agent(config=cfg)
        if model_name != "LQR":
            agent.load(f"./best_models/{model_name}/", episode=0)

        # estimate
        if isinstance(agent, LQRAgent):
            # obtain Lyapunov matrix P from agent
            P = agent.P  # assumed attribute: positive-definite matrix
            doa_frac = compute_lqr_doa_analytic(P, level, lb, ub)
        else:
            # AC critic-based Monte Carlo estimate
            critic = agent.critic_model
            critic.to(device)
            critic.eval()
            doa_frac = estimate_doa(
                level=level,
                lb=lb * 2,
                ub=ub * 2,
                critic_model=critic,
                device=device,
                n_samples=N_SAMPLES
            )
        pct = doa_frac * 100.0
        print(f"{model_name}: DoA = {pct:.2f}% of region at level={level}")
        results.append((model_name, level, pct))

    # write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "level", "doa_percent"]);
        for model_name, level, pct in results:
            writer.writerow([model_name, f"{level}", f"{pct:.4f}"])
    print(f"Saved DoA coverage to {OUTPUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
