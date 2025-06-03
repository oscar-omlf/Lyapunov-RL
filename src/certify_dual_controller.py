import numpy as np
import scipy.linalg
import torch
import dreal as d

from agents.agent_factory import AgentFactory
from util.dynamics import vanderpol_dynamics_torch
from util.dreal import is_unsat


config_lac = {
    "agent_str": "Lyapunov-AC",
    "alpha": 0.1,
    "actor_lr": 3e-3,
    "critic_lr": 2e-3,
    "dynamics_fn": vanderpol_dynamics_torch,
    "batch_size": 64,
    "num_paths_sampled": 8,
    "dt": 0.01,
    "norm_threshold": 5e-2,
    "integ_threshold": 50,
    "r1_bounds": (np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
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

config_las_lyac = {
    "agent_str": "LAS-LAC",
    "LQR": config_lqr,
    "LAC": config_lac,
    "beta": 0.5,
    "dynamics_fn": vanderpol_dynamics_torch,
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
}


agent = AgentFactory.create_agent(config=config_las_lyac)
agent.load()


def combined_check(level, scale=2., eps=0.5):
    return agent.lac_agent.trainer.check_lyapunov(level, scale, eps)

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