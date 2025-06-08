import math 
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib import cm
import torch

from util.dynamics import pendulum_dynamics_torch, pendulum_dynamics_dreal, vanderpol_dynamics_torch
from agents.lyapunov_agent import LyapunovAgent
from agents.lqr_agent import LQRAgent
# from agents.las_lyac_agent import LAS_LyACAgent

G = 9.81
L_POLE = 0.5
M_BALL = 0.15
MAX_ACTION = 1.0

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
    """
    Computes the state derivatives (vector field) for plotting.
    - policy_callable: A function that takes states and returns actions.
    - policy_expects_torch: True if policy_callable expects torch.Tensor, False if np.ndarray.
    """
    if policy_expects_torch:
        states_torch = np_to_torch(state_np_batch, device_str=torch_device_str)
        with torch.no_grad():
            actions_torch = policy_callable(states_torch)
    else: # Policy expects NumPy
        actions_np = policy_callable(state_np_batch)
        actions_torch = np_to_torch(actions_np, device_str=torch_device_str)

    if actions_torch.ndim == 1:
        if actions_torch.shape[0] == state_np_batch.shape[0]:
            actions_torch = actions_torch.unsqueeze(-1)
        else: # Scalar action, tile it
            actions_torch = actions_torch.repeat(state_np_batch.shape[0], 1 if action_dim > 0 else 0)
            if action_dim == 1: actions_torch = actions_torch.unsqueeze(-1)

    elif actions_torch.ndim == 2 and actions_torch.shape[1] != action_dim :
        raise ValueError(f"Action dimension mismatch. Expected {action_dim}, got {actions_torch.shape[1]}")

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

    DX_DY_np = get_vector_field_np(states_for_flow_np,
                                   policy_callable,
                                   torch_device_str,
                                   provided_dynamics_fn_torch,
                                   policy_expects_torch=policy_expects_torch,
                                   action_dim=action_dim)
    DX_np = DX_DY_np[:, 0].reshape(Xd_np.shape)
    DY_np = DX_DY_np[:, 1].reshape(Xd_np.shape)

    ax.streamplot(Xd_np, Yd_np, DX_np, DY_np, color='gray', linewidth=0.5,
                  density=1.2, arrowstyle='-|>', arrowsize=1.2)

def plot_lyapunov_3d(
        X_np: np.ndarray, 
        Y_np: np.ndarray, 
        Val_np: np.ndarray,
        title: str, 
        z_label: str = '$W(x)$'
        ) -> Tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(projection='3d')

    if np.all(np.isnan(Val_np)) or np.all(np.isinf(Val_np)):
        print(f"Warning: All values in Val_np for '{title}' are NaN/inf. Plotting will be empty.")
        ax.text(0.5, 0.5, 0.5, "Invalid Data (NaN/Inf)", transform=ax.transAxes, ha='center', va='center')
    else:
        min_val_offset = np.nanmin(Val_np) if np.any(np.isfinite(Val_np)) else 0
        min_val_offset -= 0.05 * abs(min_val_offset)
        surf = ax.plot_surface(X_np, Y_np, Val_np, rstride=4, cstride=4, alpha=0.8, cmap=cm.viridis,
                               vmin=np.nanpercentile(Val_np, 5), vmax=np.nanpercentile(Val_np, 95))
        ax.contour(X_np, Y_np, Val_np, levels=15, zdir='z', offset=min_val_offset, cmap=cm.viridis, linewidths=0.8)
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)

    ax.set_xlabel('Angle $\Theta$ (rad)')
    ax.set_ylabel('Angular Velocity $\dot{\Theta}$ (rad/s)')
    ax.set_zlabel(z_label)
    ax.set_title(title)
    return fig, ax

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device_name}")

config_lac_pendulum = {
    "agent_str": "Lyapunov-AC",
    "model_name": "LAC",
    "alpha": 0.2,
    "lr": 2e-3,
    "dynamics_fn": pendulum_dynamics_torch,
    "dynamics_fn_dreal": pendulum_dynamics_dreal,
    "batch_size": 64,
    "num_paths_sampled": 8,
    "dt": 0.003,
    "norm_threshold": 5e-2,
    "integ_threshold": 150,
    "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
    "actor_hidden_sizes": (5, 5),
    "critic_hidden_sizes": (20, 20),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
    "max_action": 1.0
}

print("Initializing Lyapunov-AC Agent...")
lac_agent = LyapunovAgent(config=config_lac_pendulum)

lac_agent.load(file_path='./logs/LAC/run_2/', episode=3000) 
print("Lyapunov-AC Agent loaded successfully.")

config_lqr = {
    "agent_str": "LQR",
    "environment": "InvertedPendulum",
    "discrete_discounted": True,
    "gamma": 0.99,
    "dt": 0.03,
    "g": 9.81,
    "m": 0.15,
    "l": 0.5,
    "max_action": 1.0,
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
}

print("Initializing LQR Agent...")
lqr_agent = LQRAgent(config=config_lqr)
P_lqr_np = lqr_agent.P_np

angle_plt_range = np.linspace(-math.pi, math.pi, 150)
velocity_plt_range = np.linspace(-8.0, 8.0, 150)
X_grid_np, Y_grid_np = np.meshgrid(angle_plt_range, velocity_plt_range)

states_flat_np = np.stack([X_grid_np.ravel(), Y_grid_np.ravel()], axis=-1)
states_flat_torch = np_to_torch(states_flat_np, device_str=device_name)

# Evaluate W(x) from LyAC Critic
W_grid_np = np.full_like(X_grid_np, np.nan)
try:
    with torch.no_grad():
        W_flat_torch = lac_agent.critic_model(states_flat_torch)
        # W_flat_torch = las_lac_agent.lyapunov_value(states_flat_torch)
        W_grid_np = torch_to_np(W_flat_torch.squeeze()).reshape(X_grid_np.shape)
except Exception as e:
    print(f"Error evaluating LyAC critic: {e}. W_grid_np will contain NaNs.")

# Evaluate LQR Lyapunov Function V(x) = (x-x*)^T P (x-x*)
x_star_lqr_np = lqr_agent.x_star_np # Should be np.array([0.0, 0.0])
delta_x_grid_theta = X_grid_np - x_star_lqr_np[0]
delta_x_grid_thetadot = Y_grid_np - x_star_lqr_np[1]

p11, p12 = P_lqr_np[0,0], P_lqr_np[0,1]
p21, p22 = P_lqr_np[1,0], P_lqr_np[1,1] # p12 should be equal to p21 for symmetric P
if not np.isclose(p12, p21):
    print(f"Warning: LQR P matrix is not symmetric: p12={p12}, p21={p21}")

V_lqr_grid_np = p11 * delta_x_grid_theta**2 + \
                (p12 + p21) * delta_x_grid_theta * delta_x_grid_thetadot + \
                p22 * delta_x_grid_thetadot**2

# 3D Plot of W(x) 
print("Generating 3D plot for LyAC W_learned(x)...")
fig_3d_lac, ax_3d_lac = plot_lyapunov_3d(
    X_grid_np, Y_grid_np, W_grid_np,
    title='LyAC Learned Zubov Function $W_{learned}(x)$',
    z_label='$W_{learned}(x)$'
)
if ax_3d_lac.has_data():
    ax_3d_lac.view_init(elev=25, azim=-125)
plt.savefig('./pendulum_W_lac_3d.png', dpi=300, bbox_inches='tight')
# plt.show()

# 3D Plot of LQR V(x)
print("Generating 3D plot for LQR V(x)...")
fig_3d_lqr, ax_3d_lqr = plot_lyapunov_3d(X_grid_np, Y_grid_np, V_lqr_grid_np,
                                         title='LQR Lyapunov Function $V_{LQR}(x)$',
                                         z_label='$V_{LQR}(x)$')
if ax_3d_lqr.has_data():
    ax_3d_lqr.view_init(elev=25, azim=-125)
plt.savefig('./pendulum_V_lqr_3d.png', dpi=300, bbox_inches='tight')
# plt.show()


# 2D RoA Plot
print("Generating 2D RoA plot...")
fig_2d, ax_2d = plt.subplots(figsize=(7.5, 6.5))

# Liens for LyAC policy
angle_flow_range = np.linspace(X_grid_np.min(), X_grid_np.max(), 30)
velocity_flow_range = np.linspace(Y_grid_np.min(), Y_grid_np.max(), 30)
X_flow_np, Y_flow_np = np.meshgrid(angle_flow_range, velocity_flow_range)

plot_streamlines(ax_2d, X_flow_np, Y_flow_np,
                 lac_agent.actor_model,
                 torch_device_str=device_name,
                 provided_dynamics_fn_torch=pendulum_dynamics_torch,
                 policy_expects_torch=True,
                 action_dim=1)

# Contour for LQR V(x)
alpha = 0.2
lqr_contour_val = 0.4230  # TODO: Tune this!
lqr_contour_val = math.tanh(lqr_contour_val) / alpha
print(f"LQR certified c* = {lqr_contour_val:.4f}")
ax_2d.contour(X_grid_np, Y_grid_np, V_lqr_grid_np, levels=[lqr_contour_val], linewidths=2, colors='magenta', linestyles='--')

# Contour for LyAC W(x)
lyac_contour_val = 0.0779
ax_2d.contour(X_grid_np, Y_grid_np, W_grid_np, levels=[lyac_contour_val],linewidths=2, colors='red')

# Plot Styling
ax_2d.set_title('Regions of Attraction - Inverted Pendulum')
ax_2d.set_xlabel('Angle $\Theta$ (rad)')
ax_2d.set_ylabel('Angular Velocity $\dot{\Theta}$ (rad/s)')
ax_2d.axhline(0, color='black', lw=0.5, linestyle=':')
ax_2d.axvline(0, color='black', lw=0.5, linestyle=':')
ax_2d.grid(True, linestyle=':', alpha=0.6)

legend_handles = [
    plt.Line2D([0], [0], color='magenta', linestyle='--', linewidth=2, label=f'LQR $V(x)={lqr_contour_val:.2f}$'),
    plt.Line2D([0], [0], color='red', linewidth=2, label=f'LyAC $W(x)={lyac_contour_val:.2f}$')
]
ax_2d.legend(handles=legend_handles, loc='upper right') # Use handles for custom legend

ax_2d.set_xlim(X_grid_np.min(), X_grid_np.max())
ax_2d.set_ylim(Y_grid_np.min(), Y_grid_np.max())

plt.savefig('./pendulum_roa_2d.png', dpi=300, bbox_inches='tight')
# plt.show()

print("All plots generated.")