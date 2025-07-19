import math 
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib import cm
import torch

from agents.agent_factory import AgentFactory
from config import (
    config_ldp_pendulum,
    config_lac_pendulum,
    config_lqr_pendulum
)

CFG_LDP = config_ldp_pendulum
CFG_LAC = config_lac_pendulum
CFG_LQR = config_lqr_pendulum

dynamics_fn = CFG_LAC["dynamics_fn"]


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

    print("Initializing LDP Agent...")
    ldp_agent = AgentFactory.create_agent(config=CFG_LDP)
    ldp_agent.load(file_path='best_models/LDP/0.9/', episode=0)
    CFG_LDP["beta"] = 0.9
    print("LDP Agent loaded successfully.")

    print("Initializing Lyapunov-AC Agent...")
    lac_agent = AgentFactory.create_agent(config=CFG_LAC)
    lac_agent.load(file_path='best_models/LAC/', episode=0)
    print("Lyapunov-AC Agent loaded successfully.")

    print("Initializing LQR Agent...")
    lqr_agent = AgentFactory.create_agent(config=CFG_LQR)
    P_lqr_np = lqr_agent.P_np
    alpha = CFG_LAC["alpha"]

    # Grid and Evaluation
    angle_plt_range = np.linspace(-math.pi, math.pi, 150)
    velocity_plt_range = np.linspace(-8.0, 8.0, 150)
    X_grid_np, Y_grid_np = np.meshgrid(angle_plt_range, velocity_plt_range)

    states_flat_np = np.stack([X_grid_np.ravel(), Y_grid_np.ravel()], axis=-1)
    states_flat_torch = np_to_torch(states_flat_np, device_str=device_name)

    # Evaluate W(x) from the LDP Agent
    with torch.no_grad():
        W_ldp_flat_torch = ldp_agent.get_composite_W_value(states_flat_torch)
        W_ldp_grid_np = torch_to_np(W_ldp_flat_torch.squeeze()).reshape(X_grid_np.shape)

    # Evaluate W(x) from LAC Critic
    with torch.no_grad():
        W_lac_flat_torch = lac_agent.critic_model(states_flat_torch)
        W_lac_grid_np = torch_to_np(W_lac_flat_torch.squeeze()).reshape(X_grid_np.shape)

    # Evaluate LQR Lyapunov Function V(x) and the transformed W(x)
    x_star_lqr_np = lqr_agent.x_star_np
    delta_x_flat = states_flat_np - x_star_lqr_np
    V_lqr_flat_np = np.sum((delta_x_flat @ P_lqr_np) * delta_x_flat, axis=1)
    V_lqr_grid_np = V_lqr_flat_np.reshape(X_grid_np.shape)
    
    # 3D Plots
    print("Generating 3D plots...")
    fig_3d_ldp, ax_3d_ldp = plot_lyapunov_3d(X_grid_np, Y_grid_np, W_ldp_grid_np, 'LDP Learned Zubov Function $W^{\phi}(x)$', '$W^{\phi}(x)$')
    if ax_3d_ldp.has_data(): ax_3d_ldp.view_init(elev=25, azim=-125)
    plt.savefig('./pendulum_W_ldp_3d.png', dpi=300, bbox_inches='tight')
    
    fig_3d_lac, ax_3d_lac = plot_lyapunov_3d(X_grid_np, Y_grid_np, W_lac_grid_np, 'LAC Learned Zubov Function $W(x)$', '$W(x)$')
    if ax_3d_lac.has_data(): ax_3d_lac.view_init(elev=25, azim=-125)
    plt.savefig('./pendulum_W_lac_3d.png', dpi=300, bbox_inches='tight')

    fig_3d_lqr, ax_3d_lqr = plot_lyapunov_3d(X_grid_np, Y_grid_np, V_lqr_grid_np, 'LQR Lyapunov Function $V}(x)$', '$V(x)$')
    if ax_3d_lqr.has_data(): ax_3d_lqr.view_init(elev=25, azim=-125)
    plt.savefig('./pendulum_V_lqr_3d.png', dpi=300, bbox_inches='tight')

    # 2D RoA Plot
    print("Generating 2D RoA plot...")
    fig_2d, ax_2d = plt.subplots(figsize=(7.5, 6.5))

    # Streamlines for LAC policy
    angle_flow_range = np.linspace(X_grid_np.min(), X_grid_np.max(), 30)
    velocity_flow_range = np.linspace(Y_grid_np.min(), Y_grid_np.max(), 30)
    X_flow_np, Y_flow_np = np.meshgrid(angle_flow_range, velocity_flow_range)
    plot_streamlines(ax_2d, X_flow_np, Y_flow_np, lac_agent.actor_model, device_name, dynamics_fn)

    # Contour for LQR V(x)
    lqr_v_contour_val = 1.1523
    print(f"Plotting LQR V(x) certified c* = {lqr_v_contour_val:.4f}")
    ax_2d.contour(X_grid_np, Y_grid_np, V_lqr_grid_np, levels=[lqr_v_contour_val], linewidths=2, colors='magenta', linestyles='--')

    # Contour for LAC W(x)
    lac_contour_val = 0.9593
    ax_2d.contour(X_grid_np, Y_grid_np, W_lac_grid_np, levels=[lac_contour_val], linewidths=2, colors='red')

    # Contour for LDP W(x)
    ldp_contour_val = 0.95
    ax_2d.contour(X_grid_np, Y_grid_np, W_ldp_grid_np, levels=[ldp_contour_val], linewidths=2, colors='blue')

    ax_2d.set_title('Regions of Attraction - Inverted Pendulum')
    ax_2d.set_xlabel('Angle $\Theta$ (rad)')
    ax_2d.set_ylabel('Angular Velocity $\dot{\Theta}$ (rad/s)')
    ax_2d.grid(True, linestyle=':', alpha=0.6)
    
    legend_handles = [
        plt.Line2D([0], [0], color='magenta', linestyle='--', label=f'LQR $V(x) \\leq {lqr_v_contour_val:.2f}$'),
        plt.Line2D([0], [0], color='red', label=f'LAC $W(x) \\leq {lac_contour_val:.2f}$'),
        plt.Line2D([0], [0], color='blue', label=f'LDP $W(x) \\leq {ldp_contour_val:.2f}$'),
    ]
    ax_2d.legend(handles=legend_handles, loc='upper right')
    ax_2d.set_xlim(X_grid_np.min(), X_grid_np.max())
    ax_2d.set_ylim(Y_grid_np.min(), Y_grid_np.max())
    plt.savefig('./pendulum_roa_2d.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("All plots generated.")

if __name__ == '__main__':
    main()