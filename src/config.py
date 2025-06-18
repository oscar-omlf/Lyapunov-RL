import numpy as np

from util.dynamics import (
    pendulum_dynamics_torch,
    pendulum_dynamics_np,
    pendulum_dynamics_dreal,
    compute_pendulum_reward,
    vanderpol_dynamics_torch,
    vanderpol_dynamics_dreal
)

G = 9.81
M_BALL = 0.15
L_POLE = 0.5
B_POLE = 0.1
MAX_ACTION = 1.0


config_ldp_pendulum = {
    "agent_str": "LDP",
    "model_name": "LDP",
    "environment": "InvertedPendulum",
    "max_action": 1.0,
    "beta": 0.8,
    "dynamics_fn_dreal": pendulum_dynamics_dreal,
    "dynamics_fn": pendulum_dynamics_torch,
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
    "batch_size": 128,
    "num_paths_sampled": 8,
    "norm_threshold": 5e-2,
    "integ_threshold": 500,
    "dt": 0.003,
    "actor_hidden_sizes":  (5, 5),
    "critic_hidden_sizes": (20, 20),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
    "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
    "normalize_gradients": False,
    "c_star": 1.1523,
}

config_ldp_vanderpol = {
    "agent_str": "LDP",
    "model_name": "LDP",
    "environment": "VanDerPol",
    "max_action": 1.0,
    "beta": 0.8,
    "dynamics_fn_dreal": vanderpol_dynamics_dreal,
    "dynamics_fn": vanderpol_dynamics_torch,
    "LQR": {
        "agent_str": "LQR",
        "environment": "VanDerPol",
        "discrete_discounted": False,
        "mu": 1.0,
        "max_action": 1.0,
        "R": 0.1 * np.eye(1),
        "state_space": np.zeros(2),
        "action_space": np.zeros(1),
    },
    "alpha": 0.1,
    "lr": 3e-3,
    "batch_size": 128,
    "num_paths_sampled": 8,
    "norm_threshold": 5e-2,
    "integ_threshold": 150,
    "dt": 0.01,
    "actor_hidden_sizes":  (30, 30),
    "critic_hidden_sizes": (30, 30),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
    "r1_bounds": (np.array([-2.0, -2.0]), np.array([2.0, 2.0])), 
    "normalize_gradients": True,
    "c_star": 3.9990,
}

config_lac_pendulum = {
    "agent_str": "Lyapunov-AC",
    "model_name": "LAC",
    "environment": "InvertedPendulum",
    "alpha": 0.2,
    "lr": 2e-3,
    "dynamics_fn": pendulum_dynamics_torch,
    "dynamics_fn_dreal": pendulum_dynamics_dreal,
    "batch_size": 128,
    "num_paths_sampled": 16,
    "dt": 0.003,
    "norm_threshold": 5e-2,
    "integ_threshold": 500,
    "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
    "actor_hidden_sizes": (5, 5), 
    "critic_hidden_sizes": (20, 20),
    "state_space": np.zeros(2), 
    "action_space": np.zeros(1), 
    "max_action": 1.0,
    "normalize_gradients": False
}

config_lac_vanderpol = {
    "agent_str": "Lyapunov-AC",
    "model_name": "LAC",
    "environment": "VanDerPol",
    "alpha": 0.1,
    "lr": 3e-3,
    "dynamics_fn": vanderpol_dynamics_torch,
    "dynamics_fn_dreal": vanderpol_dynamics_dreal,
    "batch_size": 128,
    "num_paths_sampled": 8,
    "dt": 0.01,
    "norm_threshold": 5e-2,
    "integ_threshold": 150,
    "r1_bounds": (np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
    "actor_hidden_sizes": (30, 30), 
    "critic_hidden_sizes": (30, 30),
    "state_space": np.zeros(2), 
    "action_space": np.zeros(1), 
    "max_action": 1.0,
    "normalize_gradients": True
}

config_td3_pendulum = {
    "agent_str": "TD3",
    "model_name": "TD3",
    "environment": "InvertedPendulum",
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
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
    "max_action": MAX_ACTION,
    "dynamics_fn": pendulum_dynamics_np,
    "rewards_fn": compute_pendulum_reward,
}

config_lqr_pendulum = {
    "agent_str": "LQR",
    "model_name": "LQR",
    "environment": "InvertedPendulum",
    "discrete_discounted": False,
    "gamma": 0.99,
    "dt": 0.003,
    "g": G,
    "m": M_BALL,
    "l": L_POLE,
    "b": B_POLE,
    "max_action": MAX_ACTION,
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
}