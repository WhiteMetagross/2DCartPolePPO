#This program evaluates a trained PPO agent on the CartPole-v1 environment.
#It can be used to evaluate the performance of a trained agent and to create 
#visualization plots of the evaluation results.

import os
import gymnasium as gym
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

from config import get_default_config

#Stable-Baselines3 PPO:
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

#This function evaluates the performance of a trained PPO agent on the CartPole-v1 environment.
def evaluate_agent(model, env_name, num_episodes=100, render=False, create_plots=False):
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    episode_rewards = []
    episode_lengths = []
    successes = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(state, deterministic=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if episode_reward >= 475:
            successes += 1

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {np.mean(episode_rewards[-10:]):.2f}")

    env.close()

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = successes / num_episodes

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of episodes: {num_episodes}")
    print(f"Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average episode length: {avg_length:.2f}")
    print(f"Success rate (reward >= 475): {success_rate:.2%}")
    print(f"Min reward: {min(episode_rewards):.2f}")
    print(f"Max reward: {max(episode_rewards):.2f}")

    if create_plots:
        create_evaluation_plots(episode_rewards, episode_lengths)

    return {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_length": avg_length,
        "success_rate": success_rate,
        "min_reward": min(episode_rewards),
        "max_reward": max(episode_rewards),
        "all_rewards": episode_rewards
    }

#This function creates evaluation plots for a trained PPO agent.
def create_evaluation_plots(episode_rewards, episode_lengths):
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    #Plot episode rewards.
    ax1.plot(episode_rewards, alpha=0.7, color='blue')
    ax1.axhline(y=475, color='red', linestyle='--', alpha=0.8, label='Success Threshold')
    rolling_mean = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
    ax1.plot(range(9, len(episode_rewards)), rolling_mean, color='orange', linewidth=2, label='10-Episode Rolling Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    #Plot reward distribution.
    ax2.hist(episode_rewards, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(x=np.mean(episode_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(episode_rewards):.2f}')
    ax2.axvline(x=475, color='orange', linestyle='--', label='Success Threshold')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    #Plot episode lengths.
    ax3.plot(episode_lengths, alpha=0.7, color='purple')
    rolling_mean_length = np.convolve(episode_lengths, np.ones(10)/10, mode='valid')
    ax3.plot(range(9, len(episode_lengths)), rolling_mean_length, color='orange', linewidth=2, label='10-Episode Rolling Average')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Episode Length')
    ax3.set_title('Episode Lengths Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    #Plot reward vs episode length. Handle zero-variance safely to avoid numpy warnings.
    std_r = float(np.std(episode_rewards))
    std_l = float(np.std(episode_lengths))
    if std_r == 0.0 or std_l == 0.0:
        reward_vs_length = float('nan')
    else:
        reward_vs_length = float(np.corrcoef(episode_rewards, episode_lengths)[0, 1])

    ax4.scatter(episode_lengths, episode_rewards, alpha=0.6, color='red')
    ax4.set_xlabel('Episode Length')
    ax4.set_ylabel('Episode Reward')
    corr_str = 'N/A' if np.isnan(reward_vs_length) else f'{reward_vs_length:.3f}'
    ax4.set_title(f'Reward vs Episode Length (Correlation: {corr_str})')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(15, 6))

    bins = np.arange(0, len(episode_rewards) + 10, 10)
    binned_rewards = []
    for i in range(len(bins) - 1):
        start_idx = bins[i]
        end_idx = min(bins[i + 1], len(episode_rewards))
        if start_idx < len(episode_rewards):
            binned_rewards.append(np.mean(episode_rewards[start_idx:end_idx]))

    ax5.bar(range(len(binned_rewards)), binned_rewards, alpha=0.7, color='teal')
    ax5.axhline(y=475, color='red', linestyle='--', alpha=0.8, label='Success Threshold')
    ax5.set_xlabel('Episode Batch (10 episodes each)')
    ax5.set_ylabel('Average Reward')
    ax5.set_title('Average Reward per 10-Episode Batch')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    success_episodes = [i for i, reward in enumerate(episode_rewards) if reward >= 475]
    success_bins = np.arange(0, len(episode_rewards) + 10, 10)
    success_counts = []
    for i in range(len(success_bins) - 1):
        start_idx = success_bins[i]
        end_idx = min(success_bins[i + 1], len(episode_rewards))
        count = sum(1 for ep in success_episodes if start_idx <= ep < end_idx)
        success_counts.append(count)

    ax6.bar(range(len(success_counts)), success_counts, alpha=0.7, color='gold')
    ax6.set_xlabel('Episode Batch (10 episodes each)')
    ax6.set_ylabel('Number of Successful Episodes')
    ax6.set_title('Successful Episodes per 10-Episode Batch')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('evaluation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

#This function loads a trained PPO model from a checkpoint file.
def load_trained_model(model_path):
    model = PPO.load(model_path, device='cpu')
    return model

#This function is the main entry point for the evaluation program.
def main():
    #Centralized configuration
    config = get_default_config()

    #Look for SB3 model checkpoints (.zip) in common locations.
    model_paths = [
        'checkpoints/best_model.zip',
        'checkpoints/final_model.zip',
        '../checkpoints/best_model.zip',
        '../checkpoints/final_model.zip'
    ]

    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        print("No trained model found. Please train a model first using train.py")
        print("Expected model locations:")
        for path in model_paths:
            print(f"  - {path}")
        return

    print(f"Loading trained model from: {model_path}")
    model = load_trained_model(model_path)
    print("Loaded trained model successfully")

    #Evaluate the agent.
    print("\nEvaluating agent performance...")
    results = evaluate_agent(model, config.env_name, num_episodes=100, render=False, create_plots=True)

    #Save results.
    with open("evaluation_results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "all_rewards"}, f, indent=4)

    print("\nDetailed results saved to evaluation_results.json")
    print("Evaluation plots saved as evaluation_plots.png and evaluation_analysis.png")

    #Render a few episodes for visual evaluation.
    render_episodes = input("\nRender 5 episodes for visual evaluation? (y/n): ").lower().strip()
    if render_episodes == 'y':
        print("Rendering 5 episodes for visual evaluation...")
        evaluate_agent(model, config.env_name, num_episodes=5, render=True, create_plots=False)


if __name__ == "__main__":
    main()
