import math
import numpy as np
import torch
import matplotlib.pyplot as plt

# Assuming your agent and utility scripts are in the correct path
from agents.lyapunov_agent import LyapunovAgent
from util.dynamics import pendulum_dynamics_torch, pendulum_dynamics_dreal

# Helper function to convert numpy arrays to torch tensors
def np_to_torch(x_np: np.ndarray, device_str: str = 'cpu') -> torch.Tensor:
    """Converts a numpy array to a PyTorch tensor."""
    return torch.from_numpy(x_np).float().to(device_str)

# Configuration for the pendulum agent, copied from your script
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

def calculate_and_plot_doa():
    """
    Calculates the area of the DoA and generates a scatter plot
    visualizing the sampled points.
    """
    # --- 1. Setup and Constants ---
    print("🚀 Starting DoA calculation and visualization...")
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device_name}")

    # State space boundaries for Inverted Pendulum
    angle_lim = (-math.pi, math.pi)
    velocity_lim = (-8.0, 8.0)
    
    # Certified level-set for the learned Zubov function W(x)
    CERTIFIED_LEVEL_C = 0.8183

    # Number of samples for accurate area calculation
    NUM_SAMPLES_FOR_AREA = 5_000_000
    # Number of samples for the visualization plot (to keep it clean)
    NUM_SAMPLES_FOR_PLOT = 50_000

    # Calculate the total area of the state space (our bounding box)
    total_state_space_area = (angle_lim[1] - angle_lim[0]) * (velocity_lim[1] - velocity_lim[0])

    # --- 2. Load the Agent ---
    print("Initializing and loading the Lyapunov-AC agent...")
    agent = LyapunovAgent(config=config_lac_pendulum)
    try:
        agent.load(file_path='logs/LAC_CEGAR/run_2/', episode=1000)
        agent.critic_model.to(device_name)
        agent.critic_model.eval()
        print("Agent loaded successfully.")
    except FileNotFoundError:
        print("\n❌ ERROR: Could not find agent model files.")
        print("Please ensure the path 'logs/LAC_CEGAR/run_2/' is correct.")
        return

    # --- 3. Monte Carlo Integration ---
    print(f"Running Monte Carlo integration with {NUM_SAMPLES_FOR_AREA:,} samples...")

    # Generate random points uniformly within the state space
    random_angles = np.random.uniform(angle_lim[0], angle_lim[1], NUM_SAMPLES_FOR_AREA)
    random_velocities = np.random.uniform(velocity_lim[0], velocity_lim[1], NUM_SAMPLES_FOR_AREA)
    states_np = np.stack([random_angles, random_velocities], axis=-1)
    states_torch = np_to_torch(states_np, device_str=device_name)

    # Evaluate the critic network W(x) for all points
    with torch.no_grad():
        w_values_torch = agent.critic_model(states_torch).squeeze()

    # --- 4. Calculate and Report Results ---
    print("Calculation complete. Reporting results...")
    
    # Create a boolean mask of "hits" (points inside the DoA)
    is_hit_mask_torch = w_values_torch < CERTIFIED_LEVEL_C
    hits = torch.sum(is_hit_mask_torch).cpu().item()

    doa_area = (hits / NUM_SAMPLES_FOR_AREA) * total_state_space_area
    doa_ratio = hits / NUM_SAMPLES_FOR_AREA
    
    # (Results printing remains the same...)
    print("\n" + "="*45)
    print("      Domain of Attraction Analysis")
    print("="*45)
    print(f"✅ Calculated DoA Area:          {doa_area:.4f} units²")
    print(f"✅ DoA Coverage Ratio:           {doa_ratio:.2%}")
    print("="*45)

    # --- 5. Generate and Save Plot ---
    print(f"\nGenerating plot with {NUM_SAMPLES_FOR_PLOT:,} sample points...")
    
    # Create a random subsample for plotting to avoid a messy plot
    plot_indices = np.random.permutation(NUM_SAMPLES_FOR_AREA)[:NUM_SAMPLES_FOR_PLOT]
    
    plot_states = states_np[plot_indices]
    w_values_numpy = w_values_torch.cpu().numpy()
    plot_w_values = w_values_numpy[plot_indices]
    plot_is_hit_mask = plot_w_values < CERTIFIED_LEVEL_C

    # Separate hits from misses for plotting
    hit_points = plot_states[plot_is_hit_mask]
    miss_points = plot_states[~plot_is_hit_mask]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot misses (outside DoA) as a light gray background
    ax.scatter(miss_points[:, 0], miss_points[:, 1], color='lightgray', s=1, label='Outside DoA')
    
    # Plot hits (inside DoA) on top in a distinct color
    ax.scatter(hit_points[:, 0], hit_points[:, 1], color='#1f77b4', s=1, label='Inside DoA')
    
    ax.set_title(f'Domain of Attraction Visualization ($W(x) < {CERTIFIED_LEVEL_C}$)')
    ax.set_xlabel('Angle $\Theta$ (rad)')
    ax.set_ylabel('Angular Velocity $\dot{\Theta}$ (rad/s)')
    ax.set_xlim(angle_lim)
    ax.set_ylim(velocity_lim)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Create a legend with larger markers
    legend = ax.legend()
    for handle in legend.legendHandles:
        handle.set_sizes([30]) # Increase marker size in legend
        
    # Save the figure
    plot_filename = 'doa_visualization.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved successfully as '{plot_filename}'")
    
    plt.close(fig) # Close the figure to free up memory


if __name__ == '__main__':
    calculate_and_plot_doa()