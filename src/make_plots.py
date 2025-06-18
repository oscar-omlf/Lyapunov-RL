import math 
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib import cm
import torch

from util.dynamics import pendulum_dynamics_torch, pendulum_dynamics_dreal, vanderpol_dynamics_torch
from agents.lyapunov_agent import LyapunovAgent
from agents.lqr_agent import LQRAgent

G = 9.81
L_POLE = 0.5
M_BALL = 0.15
B_POLE = 0.1
MAX_ACTION = 1.0
DT = 0.003

config_lac_pendulum = {
    "agent_str": "Lyapunov-AC",
    "model_name": "LAC",
    "alpha": 0.2,
    "lr": 2e-3,
    "dynamics_fn": pendulum_dynamics_torch,
    "dynamics_fn_dreal": pendulum_dynamics_dreal,
    "batch_size": 128,
    "num_paths_sampled": 8,
    "dt": DT,
    "norm_threshold": 5e-2,
    "integ_threshold": 500,
    "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
    "actor_hidden_sizes": (5, 5), 
    "critic_hidden_sizes": (20, 20),
    "state_space": np.zeros(2), 
    "action_space": np.zeros(1), 
    "max_action": 1.0,
    # "P": P,
    # "c_star": c_star_lqr
}


config_lqr_pendulum = {
    "agent_str": "LQR",
    "environment": "InvertedPendulum",
    "discrete_discounted": False,
    "gamma": 0.99,
    "dt": 0.003,
    "g": G,
    "m": M_BALL,
    "l": L_POLE,
    "b": B_POLE,
    "max_action": 1.0,
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
}


config_lqr_vanderpol = {
    "agent_str": "LQR",
    "environment": "VanDerPol",
    "discrete_discounted": False,
    "gamma": 0.99,
    "dt": 0.01,
    "mu": 1.0,
    "max_action": 1.0,
    "R": 0.1 * np.eye(1),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
}


CFG_LAC = config_lac_pendulum
CFG_LQR = config_lqr_vanderpol


def torch_to_np(x_tensor: torch.Tensor) -> np.ndarray:
    return x_tensor.cpu().detach().numpy()

def np_to_torch(x_np: np.ndarray, device_str: str = 'cpu') -> torch.Tensor:
    return torch.from_numpy(x_np).float().to(device_str)

def get_vector_field_np(state_np_batch: np.ndarray,
                        policy_callable: callable,
                        torch_device_str: str,
                        provided_dynamics_fn_torch: callable,
                        policy_expects_torch: bool = True,
                        action_dim: int = 1):
    if policy_expects_torch:
        states_torch = np_to_torch(state_np_batch, device_str=torch_device_str)
        with torch.no_grad():
            actions_torch = policy_callable(states_torch)
    else:
        actions_np = policy_callable(state_np_batch)
        actions_torch = np_to_torch(actions_np, device_str=torch_device_str)

    if actions_torch.ndim == 1:
        actions_torch = actions_torch.unsqueeze(-1)

    states_for_dynamics = np_to_torch(state_np_batch, device_str=torch_device_str)
    dxdt_torch = provided_dynamics_fn_torch(states_for_dynamics, actions_torch)
    return torch_to_np(dxdt_torch)

def plot_streamlines(ax: plt.Axes, Xd_np: np.ndarray, Yd_np: np.ndarray,
                     policy_callable: callable,
                     torch_device_str: str,
                     provided_dynamics_fn_torch: callable,
                     policy_expects_torch: bool = True,
                     action_dim: int = 1):
    states_for_flow_np = np.stack([Xd_np.ravel(), Yd_np.ravel()], axis=-1)
    DX_DY_np = get_vector_field_np(states_for_flow_np, policy_callable, torch_device_str, provided_dynamics_fn_torch, policy_expects_torch, action_dim)
    DX_np = DX_DY_np[:, 0].reshape(Xd_np.shape)
    DY_np = DX_DY_np[:, 1].reshape(Xd_np.shape)
    ax.streamplot(Xd_np, Yd_np, DX_np, DY_np, color='gray', linewidth=0.5, density=1.2, arrowstyle='-|>', arrowsize=1.2)

def plot_lyapunov_3d(X_np: np.ndarray, Y_np: np.ndarray, Val_np: np.ndarray, title: str, z_label: str) -> Tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(projection='3d')
    min_val_offset = np.nanmin(Val_np) - 0.05 * abs(np.nanmin(Val_np)) if np.any(np.isfinite(Val_np)) else 0
    surf = ax.plot_surface(X_np, Y_np, Val_np, rstride=4, cstride=4, alpha=0.8, cmap=cm.viridis, vmin=np.nanpercentile(Val_np, 5), vmax=np.nanpercentile(Val_np, 95))
    ax.contour(X_np, Y_np, Val_np, levels=15, zdir='z', offset=min_val_offset, cmap=cm.viridis, linewidths=0.8)
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    ax.set_xlabel('Angle $\Theta$ (rad)')
    ax.set_ylabel('Angular Velocity $\dot{\Theta}$ (rad/s)')
    ax.set_zlabel(z_label)
    ax.set_title(title)
    return fig, ax

def main():
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_name}")

    print("Initializing Lyapunov-AC Agent...")
    lac_agent = LyapunovAgent(config=CFG_LAC)
    lac_agent.load(file_path='best_models/LAC/', episode=0)
    print("Lyapunov-AC Agent loaded successfully.")

    print("Initializing LQR Agent...")
    lqr_agent = LQRAgent(config=CFG_LQR)
    P_lqr_np = lqr_agent.P_np
    alpha = CFG_LAC["alpha"]

    # Grid and Evaluation
    angle_plt_range = np.linspace(-math.pi, math.pi, 150)
    velocity_plt_range = np.linspace(-8.0, 8.0, 150)
    X_grid_np, Y_grid_np = np.meshgrid(angle_plt_range, velocity_plt_range)

    states_flat_np = np.stack([X_grid_np.ravel(), Y_grid_np.ravel()], axis=-1)
    states_flat_torch = np_to_torch(states_flat_np, device_str=device_name)

    # Evaluate W(x) from LAC Critic
    with torch.no_grad():
        W_flat_torch = lac_agent.critic_model(states_flat_torch)
        W_grid_np = torch_to_np(W_flat_torch.squeeze()).reshape(X_grid_np.shape)

    # Evaluate LQR Lyapunov Function V(x) and the transformed W(x)
    x_star_lqr_np = lqr_agent.x_star_np
    delta_x_flat = states_flat_np - x_star_lqr_np
    V_lqr_flat_np = np.sum((delta_x_flat @ P_lqr_np) * delta_x_flat, axis=1)
    V_lqr_grid_np = V_lqr_flat_np.reshape(X_grid_np.shape)
    
    # 3D Plots
    print("Generating 3D plots...")
    fig_3d_lac, ax_3d_lac = plot_lyapunov_3d(X_grid_np, Y_grid_np, W_grid_np, 'LAC Learned Zubov Function $W_{learned}(x)$', '$W_{learned}(x)$')
    if ax_3d_lac.has_data(): ax_3d_lac.view_init(elev=25, azim=-125)
    plt.savefig('./pendulum_W_lac_3d.png', dpi=300, bbox_inches='tight')

    fig_3d_lqr, ax_3d_lqr = plot_lyapunov_3d(X_grid_np, Y_grid_np, V_lqr_grid_np, 'LQR Lyapunov Function $V_{LQR}(x)$', '$V_{LQR}(x)$')
    if ax_3d_lqr.has_data(): ax_3d_lqr.view_init(elev=25, azim=-125)
    plt.savefig('./pendulum_V_lqr_3d.png', dpi=300, bbox_inches='tight')

    # 2D RoA Plot
    print("Generating 2D RoA plot...")
    fig_2d, ax_2d = plt.subplots(figsize=(7.5, 6.5))

    # Streamlines for LAC policy
    angle_flow_range = np.linspace(X_grid_np.min(), X_grid_np.max(), 30)
    velocity_flow_range = np.linspace(Y_grid_np.min(), Y_grid_np.max(), 30)
    X_flow_np, Y_flow_np = np.meshgrid(angle_flow_range, velocity_flow_range)
    plot_streamlines(ax_2d, X_flow_np, Y_flow_np, lac_agent.actor_model, device_name, pendulum_dynamics_torch)

    # Contour for LQR V(x)
    # lqr_v_contour_val = 1.1982
    lqr_v_contour_val = 3.9990
    print(f"Plotting LQR V(x) certified c* = {lqr_v_contour_val:.4f}")
    ax_2d.contour(X_grid_np, Y_grid_np, V_lqr_grid_np, levels=[lqr_v_contour_val], linewidths=2, colors='magenta', linestyles='--')

    # Contour for LAC W(x)
    lac_contour_val = 0.9593
    ax_2d.contour(X_grid_np, Y_grid_np, W_grid_np, levels=[lac_contour_val], linewidths=2, colors='red')

    ax_2d.set_title('Regions of Attraction - Inverted Pendulum')
    ax_2d.set_xlabel('Angle $\Theta$ (rad)')
    ax_2d.set_ylabel('Angular Velocity $\dot{\Theta}$ (rad/s)')
    ax_2d.grid(True, linestyle=':', alpha=0.6)
    
    legend_handles = [
        plt.Line2D([0], [0], color='magenta', linestyle='--', label=f'LQR $V(x) \\leq {lqr_v_contour_val:.2f}$'),
        plt.Line2D([0], [0], color='red', label=f'LAC $W(x) \\leq {lac_contour_val:.2f}$')
    ]
    ax_2d.legend(handles=legend_handles, loc='upper right')
    ax_2d.set_xlim(X_grid_np.min(), X_grid_np.max())
    ax_2d.set_ylim(Y_grid_np.min(), Y_grid_np.max())
    plt.savefig('./pendulum_roa_2d.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("All plots generated.")

if __name__ == '__main__':
    main()