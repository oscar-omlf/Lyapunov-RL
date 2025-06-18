import torch
import numpy as np
import matplotlib.pyplot as plt
from agents.lqr_agent import LQRAgent
from util.sampling import sample_in_lqr_ellipsoid_torch, sample_on_circle_boundary_torch, sample_out_of_region_torch
from src.util.doa_utils import calculate_lqr_roa_parameters

G = 9.81
L_POLE = 0.5
M_BALL = 0.15
B_POLE = 0.1

def visualize_lqr_doa_region():
    config_lqr_pendulum = {
        "agent_str": "LQR",
        "environment": "InvertedPendulum",
        "discrete_discounted": False,
        "dt": 0.03,
        "g": G,
        "m": M_BALL,
        "l": L_POLE,
        "b": B_POLE,
        "max_action": 1.0,
        "state_space": np.zeros(2),
        "action_space": np.zeros(1),
    }

    agent = LQRAgent(config=config_lqr_pendulum)
    
    c_star_lqr = 1.1522
    
    num_points_to_sample = 15000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Sampling {num_points_to_sample} points from INSIDE the LQR RoA...")
    print(f"Using P matrix:\n{agent.P_np}")
    print(f"Using certified c* value: {c_star_lqr}")
    
    L_inv, r2_radius = calculate_lqr_roa_parameters(agent.P_np, c_star_lqr, scale=1.25)
    print(r2_radius)

    interior_points = sample_in_lqr_ellipsoid_torch(
        num_samples=num_points_to_sample,
        c_star=c_star_lqr,
        L_inv=L_inv,
        device=device
    )

    interior_points = interior_points.cpu().numpy()

    exterior_points = sample_out_of_region_torch(
        num_samples=num_points_to_sample,
        lb=np.array([-2, -4]),
        ub=np.array([2, 4]),
        scale=2.0,
        device=device
    )

    exterior_points = exterior_points.cpu().numpy()

    # Plot interior points
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(interior_points[:, 0], interior_points[:, 1], s=1.5, alpha=0.5, label=f'{num_points_to_sample} Sampled Interior Points') 

    ax.set_title(f"LQR Region of Attraction ($V(x) = x^T P x \leq {c_star_lqr:.3f}$)")
    ax.set_xlabel("Angle $\Theta$ (rad)")
    ax.set_ylabel("Angular Velocity $\dot{\Theta}$ (rad/s)")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-9, 9)

    ax.grid(True, linestyle=':')
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    plt.legend()
    plt.savefig('./lqr_doa_region_visualization.png', dpi=300, bbox_inches='tight')

    print("Region visualization plot saved as 'lqr_doa_region_visualization.png'")

    # Plot exterior points
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(exterior_points[:, 0], exterior_points[:, 1], s=1.5, alpha=0.5, label=f'{num_points_to_sample} Sampled Exterior Points') 

    ax.set_title(f"LQR Region of Attraction ($V(x) = x^T P x \leq {c_star_lqr:.3f}$)")
    ax.set_xlabel("Angle $\Theta$ (rad)")
    ax.set_ylabel("Angular Velocity $\dot{\Theta}$ (rad/s)")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-9, 9)

    ax.grid(True, linestyle=':')
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    plt.legend()
    plt.savefig('./lqr_doa_border_visualization.png', dpi=300, bbox_inches='tight')

    print("Region visualization plot saved as 'lqr_doa_border_visualization.png'")

if __name__ == '__main__':
    visualize_lqr_doa_region()
