import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import gymnasium as gym

from agents.agent_factory import AgentFactory

def compare_doa(config_ac, config_lqr, theta_range=(-8, 8), theta_dot_range=(-8, 8), grid_points=200):
    env = gym.make("Pendulum-v1")

    ac_agent = AgentFactory.create_agent(config_ac, env)
    lqr_agent = AgentFactory.create_agent(config_lqr, env)
    ac_agent.load()
    lqr_agent.load()

    # Construct a grid in the (theta, theta_dot) space
    theta_vals = np.linspace(theta_range[0], theta_range[1], grid_points)
    theta_dot_vals = np.linspace(theta_dot_range[0], theta_dot_range[1], grid_points)
    THETA, THETA_DOT = np.meshgrid(theta_vals, theta_dot_vals)

    # Evaluate LQR's (candidate ?) Lyapunov function: V(x) = x^T P x
    P = lqr_agent.P
    V_grid = (P[0, 0] * THETA**2 +
              (P[0, 1] + P[1, 0]) * THETA * THETA_DOT +
              P[1, 1] * THETA_DOT**2)

    # Evaluate AC's "Lyapunov" (the critic)
    # TODO: note we don't have a Lyapunov function for the AC agent just yet but we will soon
    points_2d = np.vstack([THETA.ravel(), THETA_DOT.ravel()]).T
    W_vals = ac_agent.compute_lyapunov(points_2d)
    W_grid = W_vals.reshape(THETA.shape)

    print(f"AC Critic: min={W_grid.min():.3f}, max={W_grid.max():.3f}")

    # Thresholds for LQR and AC
    c_lqr = 1.0
    c_ac = 0.95  # TODO: Adjust this later

    # Plot the contours
    plt.figure(figsize=(10, 6))

    contour_lqr = plt.contour(THETA, THETA_DOT, V_grid, levels=[c_lqr],
                              colors='red', linewidths=2)
    plt.clabel(contour_lqr, inline=True, fontsize=10, fmt=f'LQR DoA V={c_lqr}')

    contour_ac = plt.contour(THETA, THETA_DOT, W_grid, levels=[c_ac],
                             colors='blue', linestyles='dotted', linewidths=2)
    plt.clabel(contour_ac, inline=True, fontsize=10, fmt=f'AC DoA W={c_ac}')

    # Overlay the AC's closed-loop phase portrait
    DTHETA, DTHETA_DOT = _compute_closed_loop_flow_field(ac_agent, THETA, THETA_DOT, config_lqr['g'])
    plt.streamplot(THETA, THETA_DOT, DTHETA, DTHETA_DOT, color='gray', density=1.0, arrowsize=1.0)

    plt.title("Comparison of DoA + Phase Portrait")
    plt.xlabel("theta")
    plt.ylabel("theta_dot")
    plt.grid(True)
    plt.xlim(theta_range)
    plt.ylim(theta_dot_range)

    legend_elems = [
        Line2D([0], [0], color='red', lw=2, label='LQR'),
        Line2D([0], [0], color='blue', lw=2, linestyle='dotted', label='Actor-Critic')
        # Line2D([0], [0], color='gray', lw=1, label='AC phase flow') # TODO: Figure out whether to add this or not
    ]
    plt.legend(handles=legend_elems)
    plt.show()

def _compute_closed_loop_flow_field(agent, THETA, THETA_DOT, g=10.0):
    DTHETA = np.zeros_like(THETA)
    DTHETA_DOT = np.zeros_like(THETA_DOT)
    rows, cols = THETA.shape
    for i in range(rows):
        for j in range(cols):
            theta = THETA[i, j]
            theta_dot = THETA_DOT[i, j]
            obs = np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
            action = agent.policy(obs)
            u = action[0]
            dtheta = theta_dot
            dtheta_dot = -g * np.sin(theta) + u
            DTHETA[i, j] = dtheta
            DTHETA_DOT[i, j] = dtheta_dot
    return DTHETA, DTHETA_DOT
