import math
import torch
import numpy as np
import os

from util.rk4_step import rk4_step
from agents.td3_agent import TD3Agent
from util.metrics_tracker import MetricsTracker
from config import config_td3_pendulum


DT = 0.003
MAX_ACTION = 1.0


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
    total_stabilized = 0
    total_returns = []
    total_actor_losses = []
    total_critic_losses = []

    for episode in range(NUM_EPISODES):
        ep_rewards = []
        ep_actor_losses = []
        ep_critic_losses = []

        current_state_np = np.array([
                    np.random.uniform(-math.pi, math.pi),
                    np.random.uniform(-8.0, 8.0) 
                ])
        current_state_torch = torch.as_tensor(current_state_np, dtype=torch.float32, device=DEVICE)
        
        episode_reward = 0
        episode_steps = 0
        actor_loss_ep, critic_loss_ep = None, None

        episode_stabilized = False
        
        for step in range(NUM_STEPS_PER_EPISODE):
            if total_steps_taken < initial_exploration_steps:
                action_np = np.random.uniform(-MAX_ACTION, MAX_ACTION, size=(agent.action_dim,)) 
                action_torch = torch.as_tensor(action_np, dtype=torch.float32, device=DEVICE)
            else:
                action_np = agent.policy(current_state_np, noise=False)  # Note that noise is False here
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

            if episode + 1 == NUM_EPISODES:
                print(f"Step {step+1}/{NUM_STEPS_PER_EPISODE}")
                print(f"  Current State: {current_state_np}")
                print(f"  Action: {action_np}")
                print(f"  Reward: {reward_float}")

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

            # if angle is less than 0.0005 episode stabilized
            if abs(current_state_np[0]) < 0.0005 and episode_stabilized is False:
                total_stabilized += 1
                episode_stabilized = True

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
    print(f"Episodes with stabilization: {total_stabilized} / {NUM_EPISODES} episodes")

    agent.save(file_path="best_models/TD3/")
    tracker.add_run_returns(agent_id_str, total_returns)
    tracker.add_run_losses(agent_id_str, total_actor_losses, total_critic_losses)
    tracker.save_top10_plots()

if __name__ == "__main__":
    main()