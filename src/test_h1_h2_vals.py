import torch
import numpy as np
from agents.las_td3_agent import LAS_TD3Agent
from util.dynamics import pendulum_dynamics_dreal


def get_state_for_V_level(agent, target_V_val, start_state=None):
    if start_state is None:
        start_state = torch.randn(1, agent.state_dim, device=agent.device)
    
    if torch.all(start_state == 0):
        start_state[0, 0] = 1.0

    blending_func = agent.blending_function
    
    V_start = blending_func.get_lyapunov_value(start_state)
    
    if V_start.item() < 1e-9:
        return get_state_for_V_level(agent, target_V_val, start_state + 0.1)

    scaling_factor = torch.sqrt(target_V_val / V_start)
    
    target_state = scaling_factor * start_state
    return target_state

def analyze_blending_values():
    """
    Main function to initialize the agent and check h1/h2 values.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    DT = 0.03
    PENDULUM_G = 9.81
    PENDULUM_M = 0.15
    PENDULUM_L = 0.5
    PENDULUM_B = 0.1
    MAX_ACTION_VAL = 1.0

    config = {
        "max_action": MAX_ACTION_VAL,
        "beta": 0.6,
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
    }

    print("Initializing agent and estimating c*...")
    agent = LAS_TD3Agent(config)
    c_star = agent.c_star
    blending_function = agent.blending_function
    print("-" * 50)

    points_of_interest = {
        "Deep Inside DoA": 0.01 * c_star,
        "A Bit Inside DoA": 0.5 * c_star,
        "On DoA Boundary": 1.0 * c_star,
        "Far Outside DoA": 3.0 * c_star,
    }

    print(f"{'Location':<20} | {'V(x) / c*':<10} | {'h1(x)':<10} | {'h2(x)':<10}")
    print("-" * 65)

    for name, target_v in points_of_interest.items():
        state = get_state_for_V_level(agent, target_v)
        v_norm, h1, h2 = blending_function.get_all_blending_terms(state)
        print(f"{name:<20} | {v_norm.item():<10.3f} | {h1.item():<10.4f} | {h2.item():<10.4f}")

if __name__ == "__main__":
    analyze_blending_values()
    