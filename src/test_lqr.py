import numpy as np

from agents.agent_factory import AgentFactory
from util.dynamics import pendulum_dynamics_np, vanderpol_dynamics_np
from util.rk4_step import rk4_step
from util.metrics_tracker import MetricsTracker


DT = 0.03
NUM_STEPS = 150
NUM_EPISODES = 1
THRESHOLD = 0.05

def main():
    # Agent and simulation configuration
    config_discrete_pendulum = {
        'agent_str':   'LQR',
        'environment': 'InvertedPendulum',
        'discrete_discounted': True,
        'gamma':          0.1,
        'dt':          DT,
        'g':           9.81,
        'm':           0.15,
        'l':           0.5,
        'max_action':  1.0,
        'state_space': np.zeros(2),
        'action_space':np.zeros(1),
    }
    config_continuous_pendulum = {
        'agent_str':   'LQR',
        'environment': 'InvertedPendulum',
        'discrete_discounted': False,
        'g':           9.81,
        'm':           0.15,
        'l':           0.5,
        'max_action':  1.0,
        'state_space': np.zeros(2),
        'action_space':np.zeros(1),
    }
    config_discrete_vanderpol = {
        'agent_str':   'LQR',
        'environment': 'VanDerPol',
        'discrete_discounted': True,
        'dt':          DT,
        'mu':          1.0,
        'max_action':  1.0,
        'state_space': np.zeros(2),
        'action_space':np.zeros(1),
    }
    config_continuous_vanderpol = {
        'agent_str':   'LQR',
        'environment': 'VanDerPol',
        'discrete_discounted': False,
        'mu':          1.0,
        'max_action':  1.0,
        'state_space': np.zeros(2),
        'action_space':np.zeros(1),
    }

    agent_configs = [
        config_discrete_pendulum]
        
    t = [config_continuous_pendulum,
        config_discrete_vanderpol,  
        config_continuous_vanderpol
    ]

    # Prepare metrics tracking
    tracker = MetricsTracker()

    for config in agent_configs:
        agent = AgentFactory.create_agent(config=config)
     
        env_name = config['environment']
        is_discrete_discounted_lqr = config['discrete_discounted']
        agent_id_str = f"{env_name}_{'DiscreteDiscountedLQR' if is_discrete_discounted_lqr else 'ContinuousLQR'}"
        print(f"\n--- Running: {agent_id_str} ---")

        num_stabilized_episodes = 0
        episode_returns = []

        # Select dynamics function and parameters based on environment
        if env_name == "InvertedPendulum":
            dynamics_fn = pendulum_dynamics_np
            target_state = np.array([0.0, 0.0])
            initial_state_sampler = lambda: np.array([
                np.random.uniform(-np.pi, np.pi),   
                np.random.uniform(-8.0, 8.0)
            ])

        elif env_name == "VanDerPol":
            dynamics_fn = vanderpol_dynamics_np
            target_state = np.array([0.0, 0.0])
            # Initial state randomization for VanDerPol (x1, x2)
            initial_state_sampler = lambda: np.array([
                np.random.uniform(-2.0, 2.0), 
                np.random.uniform(-2.0, 2.0)
            ])

        for ep in range(NUM_EPISODES):
            state = initial_state_sampler()

            for step in range(NUM_STEPS):
                action = agent.policy_np(state)
                next_state = rk4_step(dynamics_fn, state, action, DT)


                state = next_state
                print(state)
                if np.any(np.abs(state) > 50.0):
                    break
            
            # Check for stabilization at the end of the episode
            final_error_norm = np.linalg.norm(state - target_state)
            if final_error_norm < THRESHOLD:
                num_stabilized_episodes += 1
            
            episode_returns.append(-final_error_norm)

            if (ep + 1) % (NUM_EPISODES // 10) == 0:
                 print(f"  {agent_id_str}: Ep {ep+1}/{NUM_EPISODES} done. Stabilized so far: {num_stabilized_episodes}")


        stabilization_rate = num_stabilized_episodes / NUM_EPISODES
        print(f"Result for {agent_id_str}: Stabilization Rate = {stabilization_rate:.2f} ({num_stabilized_episodes}/{NUM_EPISODES})")
        
        tracker.add_run_returns(agent_id_str, episode_returns)

    tracker.plot_top_10_agents()


if __name__ == "__main__":
    main()
