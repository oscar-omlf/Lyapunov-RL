import torch
import numpy as np
import os

from util.dynamics import (
    pendulum_dynamics_np,
    compute_pendulum_reward,
    vanderpol_dynamics_np,
    compute_vanderpol_reward
)
from util.rk4_step import rk4_step
from agents.td3_agent import TD3Agent
from util.metrics_tracker import MetricsTracker


DT = 0.03
PENDULUM_G = 9.81
PENDULUM_M = 0.15
PENDULUM_L = 0.5
MAX_ACTION_VAL = 1.0

config_td3_pendulum = {
    "agent_str": "TD3",
    "environment": "InvertedPendulum",
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 256,
    "policy_freq": 2,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "start_episodes": 125,
    "expl_noise": 0.1,
    "actor_hidden_sizes": (256, 256),
    "critic_hidden_sizes": (256, 256),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
    "max_action": MAX_ACTION_VAL,
    "dynamics_fn": pendulum_dynamics_np,
    "rewards_fn": compute_pendulum_reward,
}

config_td3_vanderpol = {
    "agent_str": "TD3",
    "environment": "VanDerPol",
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "gamma": 0.95,
    "tau": 0.005,
    "batch_size": 256,
    "policy_freq": 3,
    "policy_noise": 0.2,
    "noise_clip": 0.5,
    "start_episodes": 200,
    "expl_noise": 0.20,
    "actor_hidden_sizes": (256, 256),
    "critic_hidden_sizes": (256, 256),
    "state_space": np.zeros(2),
    "action_space": np.zeros(1),
    "max_action": MAX_ACTION_VAL,
    "dynamics_fn": vanderpol_dynamics_np,
    "rewards_fn": compute_vanderpol_reward,
}

CFG = config_td3_pendulum

dynamics_fn = CFG["dynamics_fn"]
rewards_fn = CFG["rewards_fn"]

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    agent = TD3Agent(CFG)
    agent_id_str = CFG["agent_str"] + "_" + CFG["environment"]

    tracker = MetricsTracker()

    NUM_EPISODES = 1000
    NUM_STEPS_PER_EPISODE = 150
    PRINT_EVERY_EPISODES = 10

    initial_exploration_steps = 1000
    total_steps_taken = 0
    total_returns = []
    total_actor_losses = []
    total_critic_losses = []

    for episode in range(NUM_EPISODES):
        ep_rewards = []
        ep_actor_losses = []
        ep_critic_losses = []

        current_state_np = np.array([
                    np.random.uniform(-4, 4),   
                    np.random.uniform(-4.0, 4.0) 
                ])
        current_state_torch = torch.as_tensor(current_state_np, dtype=torch.float32, device=DEVICE)
        
        episode_reward = 0
        episode_steps = 0
        actor_loss_ep, critic_loss_ep = None, None

        for step in range(NUM_STEPS_PER_EPISODE):

            if total_steps_taken < initial_exploration_steps:
                action_np = np.random.uniform(-MAX_ACTION_VAL, MAX_ACTION_VAL, size=(agent.action_dim,)) 
                action_torch = torch.as_tensor(action_np, dtype=torch.float32, device=DEVICE)
            else:
                action_np = agent.policy(current_state_np)
                action_torch = torch.as_tensor(action_np, dtype=torch.float32, device=DEVICE)

            next_state_np = rk4_step(dynamics_fn, current_state_np, action_np, DT).squeeze()

            next_state_torch = torch.as_tensor(next_state_np, dtype=torch.float32, device=DEVICE)

            reward_float = rewards_fn(
                current_state_np,
                action_np.item()
            )
            reward_torch = torch.as_tensor([reward_float], dtype=torch.float32, device=DEVICE)
            
            episode_reward += reward_float
            episode_steps += 1
            total_steps_taken += 1

            done_bool = (step == NUM_STEPS_PER_EPISODE - 1)
            done_torch = torch.as_tensor([float(done_bool)], dtype=torch.float32, device=DEVICE)

            agent.add_transition((
                current_state_torch.cpu(),
                action_torch.cpu(), 
                reward_torch.cpu(), 
                next_state_torch.cpu(), 
                done_torch.cpu()
            ))

            actor_loss, critic_loss = None, None
            if total_steps_taken > initial_exploration_steps:
                actor_loss, critic_loss = agent.update()


            if actor_loss is not None: ep_actor_losses.append(actor_loss)

            if critic_loss is not None:
                ep_critic_losses.append(critic_loss)
            elif total_steps_taken > initial_exploration_steps:
                    print('error here probably', step, total_steps_taken)
                    
            ep_rewards.append(reward_float)

            current_state_torch = next_state_torch
            current_state_np = next_state_np

            if step % 50 == 0 and (episode + 1) % PRINT_EVERY_EPISODES == 0 :
                with torch.no_grad():
                    print(f"  Ep {episode+1}, Step {step+1}:"
                          f"State: [{current_state_torch[0].item():.2f}, {current_state_torch[1].item():.2f}], "
                          f"Action: {action_np.item() if action_np.ndim > 0 else action_np:.2f}, Reward: {reward_float:.2f}")
                    if actor_loss is not None and critic_loss is not None:
                        print(f"    Losses A: {actor_loss:.4f}, C: {critic_loss:.4f}")
            if done_bool:
                break

        total_returns.append(episode_reward)
        if ep_actor_losses:
            total_actor_losses.append(np.mean(ep_actor_losses))
        if ep_critic_losses:
            total_critic_losses.append(np.mean(ep_critic_losses))

        if (episode + 1) % PRINT_EVERY_EPISODES == 0:
            print(f"Episode {episode+1}/{NUM_EPISODES} | Steps: {episode_steps} | Reward: {episode_reward:.2f}")

    print("Training finished.")
    
    agent.save(file_path="best_models/TD3/")
    tracker.add_run_returns(agent_id_str, total_returns)
    tracker.add_run_losses(agent_id_str, total_actor_losses, total_critic_losses)
    tracker.save_top10_plots()

if __name__ == "__main__":
    main()