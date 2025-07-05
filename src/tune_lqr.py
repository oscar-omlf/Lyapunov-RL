#!/usr/bin/env python3
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from util.rk4_step import rk4_step
from util.dynamics import pendulum_dynamics_np
from agents.lqr_agent import LQRAgent
from config import config_lqr_pendulum

# Simulation constants
DT = 0.003                # integration step size
NUM_EPISODES = 200        # episodes per candidate
NUM_STEPS = 200           # steps per episode
WINDOW = 30               # last-window size for cost evaluation

print(f'Number of CPU cores in use: {os.cpu_count()}')

def evaluate_candidate(candidate, num_episodes=NUM_EPISODES,
                       num_steps=NUM_STEPS, window=WINDOW):
    """
    Evaluate a candidate [q1, q2, r] by simulating the pendulum dynamics
    under LQR control and computing the negative average state norm over
    the last `window` steps in each episode.
    """
    q1, q2, r_val = candidate
    # Ensure positive-definite
    q1 = max(q1, 1e-3)
    q2 = max(q2, 1e-3)
    r_val = max(r_val, 1e-3)

    # Build LQR config based on base pendulum config
    cfg = dict(config_lqr_pendulum)
    cfg['Q'] = np.diag([q1, q2])
    cfg['R'] = np.array([[r_val]])

    # Instantiate the LQR agent; config_lqr_pendulum provides required spaces
    agent = LQRAgent(cfg)

    total_cost = 0.0
    for _ in range(num_episodes):
        # Randomize initial state: [theta, theta_dot]
        state = np.array([
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-8.0, 8.0)
        ], dtype=np.float32)

        state_norms = []
        for t in range(num_steps):
            # Get LQR action as numpy array
            action = agent.policy_np(state)
            action = np.atleast_1d(action)

            # Integrate one RK4 step, passing array action
            next_state = rk4_step(pendulum_dynamics_np, state, action, DT).squeeze()

            # Record norm in the last window
            if t >= num_steps - window:
                state_norms.append(np.linalg.norm(next_state, ord=2))

            state = next_state

        episode_cost = np.mean(state_norms)
        total_cost += episode_cost

    average_cost = total_cost / num_episodes
    # We negate because our search expects maximization
    return -average_cost


def parallel_evaluate(candidates):
    """Evaluate a list of candidate arrays in parallel."""
    with ProcessPoolExecutor() as executor:
        fitness = list(executor.map(evaluate_candidate, candidates))
    return np.array(fitness)


def bees_algorithm(n=50, m=20, e=10, nep=20, nsp=15,
                   ngh=0.1, iterations=50):
    """
    Bees Algorithm to optimize [q1, q2, r] for LQR.
    Search bounds: q1∈[1e-3,100], q2∈[1e-3,10], r∈[1e-3,2]
    """
    bounds = np.array([[1e-3, 100],
                       [1e-3, 10],
                       [1e-3, 2]])
    dim = bounds.shape[0]

    # Initialize scout bees
    population = np.random.rand(n, dim)
    for i in range(dim):
        population[:, i] = bounds[i, 0] + population[:, i] * (bounds[i, 1] - bounds[i, 0])

    # Initial evaluation
    fitness = parallel_evaluate(population)
    best_idx = np.argmax(fitness)
    best_candidate = population[best_idx]
    best_reward = fitness[best_idx]

    for itr in range(iterations):
        # Sort by fitness descending
        idx_sorted = np.argsort(-fitness)
        population = population[idx_sorted]
        fitness = fitness[idx_sorted]

        new_pop = []
        # Neighborhood search around top m
        for i in range(m):
            base = population[i]
            recruits = nep if i < e else nsp
            # Generate local recruits
            local = []
            for _ in range(recruits):
                perturb = (np.random.rand(dim) * 2 - 1) * ngh * (bounds[:,1] - bounds[:,0])
                cand = np.clip(base + perturb, bounds[:,0], bounds[:,1])
                local.append(cand)
            local = np.array(local)
            local_fit = parallel_evaluate(local)
            idx_loc = np.argmax(local_fit)
            new_pop.append(local[idx_loc])

            if local_fit[idx_loc] > best_reward:
                best_reward = local_fit[idx_loc]
                best_candidate = local[idx_loc]

        # Scouts for the rest
        scouts = np.random.rand(n-m, dim)
        for i in range(dim):
            scouts[:, i] = bounds[i, 0] + scouts[:, i] * (bounds[i, 1] - bounds[i, 0])
        new_pop.extend(list(scouts))

        population = np.array(new_pop)
        fitness = parallel_evaluate(population)

        print(f"Iteration {itr+1}/{iterations}, best fitness: {best_reward:.6f}")

    return best_candidate, best_reward


if __name__ == "__main__":
    best_candidate, best_reward = bees_algorithm(iterations=30)
    print("\nBest candidate (q1, q2, r):", best_candidate)
    print("Best fitness (negative avg state norm):", best_reward)

    # Display final LQR gain
    q1, q2, r_val = best_candidate
    final_cfg = dict(config_lqr_pendulum)
    final_cfg['Q'] = np.diag([q1, q2])
    final_cfg['R'] = np.array([[r_val]])
    final_agent = LQRAgent(final_cfg)
    print("Final LQR gain K:", final_agent.K)
