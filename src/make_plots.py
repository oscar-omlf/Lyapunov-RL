import numpy as np
from numpy import cos, sin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import torch
from torch import nn

from util.dynamics import original_pendulum_dynamics
from agents.lyapunov_agent import LyapunovACAgent
from agents.lqr_agent import LQRAgent


G = 9.81
L_POLE = 0.5
M_BALL = 0.15
B_FRICTION = 0.1
MAX_TORQUE_ACTION = 1.0


G_LQR = 9.81
L_POLE_LQR = 0.5
M_BALL_LQR = 0.15
B_FRICTION_LQR = 0.1



def torch_to_np(x_tensor: torch.Tensor) -> np.ndarray:
    return x_tensor.cpu().detach().numpy()

def np_to_torch(x_np: np.ndarray, device_str: str = 'cpu') -> torch.Tensor:
    return torch.from_numpy(x_np).float().to(device_str)

def pendulum_dynamics_np(state_np_batch, action_np_batch):
    th = state_np_batch[:, 0]
    thdot = state_np_batch[:, 1]
    
    newthdot = thdot # dx1/dt = x2
    newthddot = (G / L_POLE) * np.sin(th) - \
                (B_FRICTION / (M_BALL * L_POLE**2)) * thdot + \
                action_np_batch / (M_BALL * L_POLE**2) # dx2/dt = ...
    
    return np.stack((newthdot, newthddot), axis=1)

def pendulum_dynamics_for_plot_np(state_np_batch: np.ndarray,
                                  actor_model_torch: nn.Module,
                                  torch_device: torch.device,
                                  max_torque: float,
                                  provided_dynamics_fn_torch: callable):
    states_torch = np_to_torch(state_np_batch, device_str=str(torch_device))
    
    with torch.no_grad():
        actions_torch = actor_model_torch(states_torch)
    
    # Ensure actions are correctly shaped and clamped if necessary by the dynamics_fn
    # The provided_dynamics_fn_torch will handle clamping if it's part of its definition
    
    dxdt_torch = provided_dynamics_fn_torch(states_torch, actions_torch)
    return torch_to_np(dxdt_torch)


def plot_streamlines(ax: plt.Axes, Xd_np: np.ndarray, Yd_np: np.ndarray,
                     system_dynamics_fn_np: callable,
                     actor_model_torch: nn.Module,
                     torch_device: torch.device,
                     max_torque_val: float,
                     provided_dynamics_fn_torch: callable):
    """Plots streamlines using the actor model and provided dynamics."""
    
    # Wrapper for system_dynamics_fn_np to match expected signature by streamplot if needed
    # Or, call directly by constructing states
    
    states_for_flow_np = np.stack([Xd_np.ravel(), Yd_np.ravel()], axis=-1)
    
    DX_DY_np = system_dynamics_fn_np(states_for_flow_np, actor_model_torch, torch_device, max_torque_val, provided_dynamics_fn_torch)
    DX_np = DX_DY_np[:, 0].reshape(Xd_np.shape)
    DY_np = DX_DY_np[:, 1].reshape(Xd_np.shape)
    
    ax.streamplot(Xd_np, Yd_np, DX_np, DY_np, color='gray', linewidth=0.5,
                  density=1.2, arrowstyle='-|>', arrowsize=1.2)

def plot_lyapunov_3d(X_np: np.ndarray, Y_np: np.ndarray, V_np: np.ndarray, title: str) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X_np, Y_np, V_np, rstride=4, cstride=4, alpha=0.8, cmap=cm.viridis)
    ax.contour(X_np, Y_np, V_np, levels=15, zdir='z', offset=np.min(V_np) - 0.05, cmap=cm.viridis, linewidths=0.8)
    
    ax.set_xlabel('Angle $\Theta$ (rad)')
    ax.set_ylabel('Angular Velocity $\dot{\Theta}$ (rad/s)')
    ax.set_zlabel('$W(x)$') # Critic output is W(x)
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    return fig, ax


device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device_name}")
device = torch.device(device_name)


config_lac = {
    "agent_str": "Lyapunov-AC",
    "alpha": 0.2,
    "actor_lr": 2e-3,
    "critic_lr": 2e-3,
    "dynamics_fn": original_pendulum_dynamics,
    "batch_size": 64,
    "num_paths_sampled": 8,
    "dt": 0.003,
    "norm_threshold": 5e-2,
    "integ_threshold": 150,
    "r1_bounds": (np.array([-2.0, -4.0]), np.array([2.0, 4.0])),
    "actor_hidden_sizes": (5, 5),
    "critic_hidden_sizes": (20, 20),
    "state_space": 2,
    "action_space": 1,
}

print("Loading Lyapunov-AC Agent...")
lac_agent = LyapunovACAgent(config=config_lac)
lac_agent.load()

config_lqr = {
    "agent_str": "LQR",
    "g": G_LQR,
    "l_pole": L_POLE_LQR,
    "m_ball": M_BALL_LQR,
    "b_friction": B_FRICTION_LQR,
}

print("Initializing LQR Agent...")
lqr_agent_instance = LQRAgent(config=config_lqr)
P_lqr_np = lqr_agent_instance.P


angle_plt_range = np.linspace(-2.5, 2.5, 150)
velocity_plt_range = np.linspace(-8.0, 8.0, 150)
X_grid_np, Y_grid_np = np.meshgrid(angle_plt_range, velocity_plt_range)

states_flat_for_torch_np = np.stack([X_grid_np.ravel(), Y_grid_np.ravel()], axis=-1)
states_flat_torch = np_to_torch(states_flat_for_torch_np, device_str=device_name)

with torch.no_grad():
    W_flat_torch = lac_agent.critic_model(states_flat_torch)
W_grid_np = torch_to_np(W_flat_torch).reshape(X_grid_np.shape)

# LQR: V(x) = x^T P x
p11, p12, p22 = P_lqr_np[0,0], P_lqr_np[0,1], P_lqr_np[1,1]
V_lqr_grid_np = p11 * X_grid_np**2 + 2 * p12 * X_grid_np * Y_grid_np + p22 * Y_grid_np**2

print("Generating 3D plot for W(x)...")
fig_3d, ax_3d = plot_lyapunov_3d(X_grid_np, Y_grid_np, W_grid_np, title='Zubov Function $W(x)$ - Lyapunov-AC')
ax_3d.view_init(elev=25, azim=-125)
plt.savefig('./pendulum_W_lac_3d.png', dpi=300, bbox_inches='tight')
plt.show()

print("Generating 2D RoA plot...")
fig_2d, ax_2d = plt.subplots(figsize=(7.5, 6.5))

angle_flow_range = np.linspace(angle_plt_range.min(), angle_plt_range.max(), 25)
velocity_flow_range = np.linspace(velocity_plt_range.min(), velocity_plt_range.max(), 25)
X_flow_np, Y_flow_np = np.meshgrid(angle_flow_range, velocity_flow_range)
plot_streamlines(ax_2d, X_flow_np, Y_flow_np, pendulum_dynamics_for_plot_np, lac_agent.actor_model, device, MAX_TORQUE_ACTION, original_pendulum_dynamics)

# Contour for LQR DoA
lqr_doa_contour_level = [0.0384] # Default from paper
ax_2d.contour(X_grid_np, Y_grid_np, V_lqr_grid_np, levels=lqr_doa_contour_level, linewidths=2, colors='m', linestyles='--')

# Contour for LyAC
lac_doa_contour_level = [0.7] # From paper
ax_2d.contour(X_grid_np, Y_grid_np, W_grid_np, levels=lac_doa_contour_level, linewidths=2, colors='r')

# Plot Styling
ax_2d.set_title('Region of Attraction - Inverted Pendulum')
ax_2d.set_xlabel('Angle $\Theta$ (rad)')
ax_2d.set_ylabel('Angular Velocity $\dot{\Theta}$ (rad/s)')
ax_2d.axhline(0, color='black', lw=0.5, linestyle=':')
ax_2d.axvline(0, color='black', lw=0.5, linestyle=':')
ax_2d.grid(True, linestyle=':', alpha=0.6)

legend_handles = [
    plt.Line2D([0], [0], color='m', linestyle='--', linewidth=2),
    plt.Line2D([0], [0], color='r', linewidth=2)
]
ax_2d.legend(legend_handles, ['LQR', 'Lyapunov-AC'], loc='upper right')

ax_2d.set_xlim(angle_plt_range.min(), angle_plt_range.max())
ax_2d.set_ylim(velocity_plt_range.min(), velocity_plt_range.max())

plt.savefig('./pendulum_roa_2d.png', dpi=300, bbox_inches='tight')
plt.show()

print("All plots generated.")