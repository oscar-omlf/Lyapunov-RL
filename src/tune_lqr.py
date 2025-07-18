import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from util.rk4_step import rk4_step
from util.dynamics import pendulum_dynamics_np, compute_pendulum_reward
from agents.lqr_agent import LQRAgent
from config import config_lqr_pendulum

DT = 0.003
NUM_EPISODES = 200
NUM_STEPS = 200

print(f'Number of CPU cores in use: {os.cpu_count()}')

def evaluate_candidate(candidate, num_episodes=NUM_EPISODES,
                       num_steps=NUM_STEPS):
    q1, q2, r_val = candidate
    # Make it positive-definite
    q1 = max(q1, 1e-3)
    q2 = max(q2, 1e-3)
    r_val = max(r_val, 1e-3)

    cfg = dict(config_lqr_pendulum)
    cfg['Q'] = np.diag([q1, q2])
    cfg['R'] = np.array([[r_val]])

    agent = LQRAgent(cfg)

    total_reward = 0.0
    for _ in range(num_episodes):
        state = np.array([
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-8.0, 8.0)
        ], dtype=np.float32)

        rewards = []
        for t in range(num_steps):
            action = agent.policy_np(state)
            action = np.atleast_1d(action)

            # compute reward from current state and action
            reward = compute_pendulum_reward(state, action.item())
            rewards.append(reward)

            # step dynamics
            next_state = rk4_step(pendulum_dynamics_np, state, action, DT).squeeze()
            state = next_state

        # mean reward for this episode
        episode_reward = np.mean(rewards)
        total_reward += episode_reward

    # average reward across episodes
    average_reward = total_reward / num_episodes
    return average_reward


def parallel_evaluate(candidates):
    with ProcessPoolExecutor() as executor:
        fitness = list(executor.map(evaluate_candidate, candidates))
    return np.array(fitness)


def bees_algorithm(n=50, m=20, e=10, nep=20, nsp=15,
                   ngh=0.1, iterations=50):
    bounds = np.array([[1e-3, 100],
                       [1e-3, 10],
                       [1e-3, 2]])
    dim = bounds.shape[0]

    # initialize population
    population = np.random.rand(n, dim)
    for i in range(dim):
        population[:, i] = bounds[i, 0] + population[:, i] * (bounds[i, 1] - bounds[i, 0])

    fitness = parallel_evaluate(population)
    best_idx = np.argmax(fitness)
    best_candidate = population[best_idx]
    best_fitness = fitness[best_idx]

    for itr in range(iterations):
        # sort population by fitness
        idx_sorted = np.argsort(-fitness)
        population = population[idx_sorted]
        fitness = fitness[idx_sorted]

        new_pop = []
        for i in range(m):
            base = population[i]
            recruits = nep if i < e else nsp
            local = []
            for _ in range(recruits):
                perturb = (np.random.rand(dim) * 2 - 1) * ngh * (bounds[:,1] - bounds[:,0])
                cand = np.clip(base + perturb, bounds[:,0], bounds[:,1])
                local.append(cand)
            local = np.array(local)
            local_fit = parallel_evaluate(local)
            idx_loc = np.argmax(local_fit)
            new_pop.append(local[idx_loc])

            # update global best
            if local_fit[idx_loc] > best_fitness:
                best_fitness = local_fit[idx_loc]
                best_candidate = local[idx_loc]

        # scouts
        scouts = np.random.rand(n-m, dim)
        for i in range(dim):
            scouts[:, i] = bounds[i, 0] + scouts[:, i] * (bounds[i, 1] - bounds[i, 0])
        new_pop.extend(list(scouts))

        population = np.array(new_pop)
        fitness = parallel_evaluate(population)

        print(f"Iteration {itr+1}/{iterations}, best fitness (mean reward): {best_fitness:.6f}")

    return best_candidate, best_fitness


if __name__ == "__main__":
    best_candidate, best_fitness = bees_algorithm(iterations=30)
    print("Best candidate (q1, q2, r):", best_candidate)
    print("Best fitness (mean reward):", best_fitness)

    # build final agent and display K
    q1, q2, r_val = best_candidate
    final_cfg = dict(config_lqr_pendulum)
    final_cfg['Q'] = np.diag([q1, q2])
    final_cfg['R'] = np.array([[r_val]])
    final_agent = LQRAgent(final_cfg)
    print("Final LQR gain K:", final_agent.K)
