import numpy as np
import scipy.linalg
import torch
import dreal as d

from agents.las_lyapunov_agent import LAS_LyapunovAgent
from util.dynamics import pendulum_dynamics_torch, pendulum_dynamics_dreal
from util.dreal import is_unsat


config = {
    "model_name": "LAS_LyapunovAC_Pendulum",
    "max_action": 1.0,
    "beta": 0.6,
    "dynamics_fn": pendulum_dynamics_torch,
    "dynamics_fn_dreal": pendulum_dynamics_dreal,
    "LQR": {
        "agent_str": "LQR",
        "environment": "InvertedPendulum",
        "discrete_discounted": False,
        "g": 9.81,
        "m": 0.15,
        "l": 0.5,
        "b": 0.1,
        "max_action": 1.0,
        "state_space": np.zeros(2),
        "action_space": np.zeros(1),
    },
    "alpha": 0.2,
    "lr": 2e-3,
    "batch_size": 256,
    "num_paths_sampled": 8,
    "norm_threshold": 5e-2,
    "integ_threshold": 500,
    "expl_noise": 0.1,
    "dt": 0.003,
    "actor_hidden_sizes": (5, 5), 
    "critic_hidden_sizes": (20, 20),
    "state_space": np.zeros(2), 
    "action_space": np.zeros(1), 
    "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
}

agent = LAS_LyapunovAgent(config=config)
agent.load("logs/LAS_LyapunovAC/run_0/", episode=3000)

def combined_check(level, scale=2., eps=0.5):
    is_verified, ce_model = agent.trainer.check_lyapunov_with_ce(level, scale, eps)

    if is_verified:
        return None, None
    else:
        return ce_model, ce_model

LEVEL_INIT = 0.95

def bisection(check_fun, c_max=0.95, tol=1e-3, it_max=40):
    hi_fail = c_max
    while True:
        r1, r2 = check_fun(hi_fail, eps=0.5)
        if is_unsat(r1) and is_unsat(r2):
            break
        hi_fail *= 0.5
        if hi_fail < 1e-6:
            print("No certificate even at c = 0.")
            return 0.0
        print(hi_fail)

    lo_pass = hi_fail
    hi_fail = hi_fail * 2.0

    for _ in range(it_max):
        if hi_fail - lo_pass < tol:
            break
        mid = 0.5 * (lo_pass + hi_fail)
        r1, r2 = check_fun(mid)
        ok = is_unsat(r1) and is_unsat(r2)
        print(f"c={mid:.5f}  ok={ok}")
        if ok:
            lo_pass = mid
        else:
            hi_fail = mid
    return lo_pass

c_star_combined = bisection(combined_check, LEVEL_INIT)
print(f"Combined Policy Certified c* = {c_star_combined:.4f}")
