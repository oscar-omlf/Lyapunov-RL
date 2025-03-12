import numpy as np
import gymnasium as gym
from agents.lqr_agent import LQRAgent

def evaluate_candidate(candidate, num_episodes=5, render=False):
    """
    Evaluate a candidate [q1, q2, r] by instantiating an LQRAgent with the given Q and R.
    The candidate’s performance is measured by running several episodes with random initial conditions.
    Instead of computing a custom cost, we accumulate the environment's reward over each episode.
    The function returns the average reward over the episodes (higher is better).
    """
    q1, q2, r_val = candidate
    q1 = max(q1, 1e-3)
    q2 = max(q2, 1e-3)
    r_val = max(r_val, 1e-3)
    
    config = {
        'g': 10.0,
        'Q': np.diag([q1, q2]),
        'R': np.array([[r_val]])
    }
    
    agent = LQRAgent(config)
    
    env = gym.make('Pendulum-v1', render_mode=None, g=10.0)
    
    total_reward = 0.0
    num_steps = 200

    for episode in range(num_episodes):
        obs, _ = env.reset()
        
        episode_reward = 0.0
        for t in range(num_steps):
            action = agent.policy(obs)
            u = action[0]
            obs, reward, _, _, _ = env.step([u])
            episode_reward += reward
            
            if render:
                env.render()
        
        total_reward += episode_reward

    env.close()
    return total_reward / num_episodes

def bees_algorithm(n=50, m=20, e=10, nep=20, nsp=15, ngh=0.1, iterations=50):
    """
    Implements the Bees Algorithm to optimize the LQR weighting matrices for the LQRAgent.
    The search space for candidate [q1, q2, r] is:
        q1 ∈ [0, 100], q2 ∈ [0, 10], r ∈ [0, 2]
    This version maximizes the average reward.
    """
    bounds = np.array([[0, 100],
                       [0, 10],
                       [0, 2]])
    dim = bounds.shape[0]
    
    # Initialize n scout bees randomly in the search space.
    population = np.random.rand(n, dim)
    for i in range(dim):
        population[:, i] = bounds[i, 0] + population[:, i] * (bounds[i, 1] - bounds[i, 0])
    
    # Evaluate initial population (higher reward is better)
    fitness = np.array([evaluate_candidate(candidate) for candidate in population])
    best_reward = np.max(fitness)
    best_candidate = population[np.argmax(fitness)]
    
    for itr in range(iterations):
        # Sort population by fitness in descending order (higher reward is better)
        sorted_indices = np.argsort(-fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]
        
        new_population = []
        # For the best m candidates, perform a local neighborhood search.
        for i in range(m):
            base_candidate = population[i]
            recruits = nep if i < e else nsp
            local_candidates = []
            for j in range(recruits):
                # Generate a candidate in the neighborhood of base_candidate.
                perturbation = (np.random.rand(dim) * 2 - 1) * ngh * (bounds[:, 1] - bounds[:, 0])
                candidate_local = base_candidate + perturbation
                candidate_local = np.clip(candidate_local, bounds[:, 0], bounds[:, 1])
                local_candidates.append(candidate_local)
            local_candidates = np.array(local_candidates)
            local_fitness = np.array([evaluate_candidate(candidate) for candidate in local_candidates])
            best_local_index = np.argmax(local_fitness)
            best_local_candidate = local_candidates[best_local_index]
            best_local_reward = local_fitness[best_local_index]
            new_population.append(best_local_candidate)
            
            if best_local_reward > best_reward:
                best_reward = best_local_reward
                best_candidate = best_local_candidate
        
        # Generate the remaining (n - m) scout bees randomly.
        remaining = n - m
        scouts = np.random.rand(remaining, dim)
        for i in range(dim):
            scouts[:, i] = bounds[i, 0] + scouts[:, i] * (bounds[i, 1] - bounds[i, 0])
        new_population.extend(list(scouts))
        
        population = np.array(new_population)
        fitness = np.array([evaluate_candidate(candidate) for candidate in population])
        
        print(f"Iteration {itr+1}/{iterations}, best average reward: {best_reward}")
    
    return best_candidate, best_reward

if __name__ == "__main__":
    best_candidate, best_reward = bees_algorithm(iterations=20)
    print("\nBest candidate parameters found (q1, q2, r):", best_candidate)
    print("Best average reward:", best_reward)
    
    q1, q2, r_val = best_candidate
    final_config = {
        'g': 10.0,
        'Q': np.diag([q1, q2]),
        'R': np.array([[r_val]])
    }
    final_agent = LQRAgent(final_config)
    print("Final computed LQR gain K:", final_agent.K)
