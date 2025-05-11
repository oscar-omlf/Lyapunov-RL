import numpy as np

from agents.agent_factory import AgentFactory
from util.dynamics import pendulum_dynamics_np, compute_pendulum_reward
from util.rk4_step import rk4_step
from util.metrics_tracker import MetricsTracker


def main():
    # Agent and simulation configuration
    config = {
        'agent_str':   'LQR',
        'g':           9.81,
        'm':           0.15,
        'l':           0.5,
        'max_action':  1.0,
        'state_space': np.zeros(2),
        'action_space':np.zeros(1),
    }
    agent = AgentFactory.create_agent(config=config)

    # Simulation parameters
    dt = 0.003
    num_steps = 1000
    num_episodes = 1000
    threshold = 0.05  # stabilization criterion on state norm

    # Prepare metrics tracking
    tracker = MetricsTracker()
    returns = []
    final_states = []

    for ep in range(num_episodes):
        # Random initial state: angle in [-π,π], velocity in [-8,8]
        theta0 = np.random.uniform(-np.pi, np.pi)
        vel0   = np.random.uniform(-1.0, 1.0)
        state  = np.array([theta0, vel0], dtype=np.float64)
        total_reward = 0.0

        for _ in range(num_steps):
            # Control and rollout
            raw_action = agent.policy(state)
            u = float(raw_action[0])
            r = compute_pendulum_reward(state, u)
            total_reward += r
            state = rk4_step(pendulum_dynamics_np, state, u, dt)

        # Store episode results
        returns.append(total_reward)
        final_states.append(state.copy())

    # Record and plot returns
    tracker.add_run_returns('LQR', returns)
    tracker.plot_top_10_agents()

    # Compute and print stabilization proportion
    norms = np.linalg.norm(final_states, axis=1)
    prop_stable = np.mean(norms < threshold)
    print(f"Stabilization proportion (||state||<{threshold}): {prop_stable:.3f}")


if __name__ == "__main__":
    main()
