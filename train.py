#This program trains a CartPole-v1 agent using the PPO algorithm from Stable-Baselines3.
#It logs training progress and saves checkpoints.

import os
import json
import time
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm

#Stable-Baselines3 PPO:
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


#This function saves the model checkpoint.
def save_checkpoint(model, epoch, reward, filepath, config):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    path = filepath[:-4] if filepath.endswith('.pth') else filepath
    model.save(path)

#This function evaluates the agent's performance.
def evaluate_agent(model, env_name, num_episodes=10):
    eval_env = Monitor(gym.make(env_name))
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=num_episodes, deterministic=False)
    eval_env.close()
    return mean_reward

#This function is the main training function.
def main():
    from config import get_default_config
    config = get_default_config()
    #For CartPole with MlpPolicy, CPU training is typically faster than GPU per SB3 guidance.
    device = 'cpu'

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print("Using device: CPU")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Starting PPO (Stable-Baselines3) training on {config.env_name}")
    print(f"Total timesteps: {config.total_timesteps:,}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    envs = make_vec_env(config.env_name, n_envs=config.num_envs, seed=config.seed)

    policy_kwargs = dict(net_arch=list(config.policy_hidden_dims))

    #Initialize the PPO model.
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=config.seed,
        device=device
    )

    best_avg_reward = -float("inf")
    update = 0
    start_time = time.time()

    reward_history = []
    eval_rewards = []
    steps = []
    eval_steps = []
    no_improvement_count = 0

    log_file = open('logs/training_log.txt', 'w')
    log_file.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Environment: {config.env_name}\n")
    log_file.write(f"Device: {device}\n")
    log_file.write(f"Num environments: {config.num_envs}\n")
    log_file.write(f"Config: {config.get_dict()}\n\n")

    #Training loop.
    try:
        while model.num_timesteps < config.total_timesteps:
            chunk_steps = config.n_steps * max(1, config.num_envs)
            model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False, progress_bar=False)
            update += 1

            if update % config.log_interval == 0:
                elapsed_time = time.time() - start_time
                fps = int(model.num_timesteps / elapsed_time) if elapsed_time > 0 else 0
                last_reward = reward_history[-1] if reward_history else 0.0
                log_msg = (
                    f"Update {update:4d} | Step {model.num_timesteps:7d} | "
                    f"Reward {last_reward:7.2f} | PL {0.0:6.3f} | VL {0.0:6.3f} | "
                    f"LR {config.learning_rate:.2e} | FPS {fps}"
                )
                print(log_msg)
                log_file.write(log_msg + "\n")
                log_file.flush()
                steps.append(model.num_timesteps)

            #Evaluation.
            if update % config.eval_interval == 0:
                avg_reward = evaluate_agent(model, config.env_name)
                eval_rewards.append(avg_reward)
                eval_steps.append(update)
                reward_history.append(avg_reward)

                eval_msg = f"Evaluation | Update {update:4d} | Avg Reward: {avg_reward:7.2f}"
                print(eval_msg)
                log_file.write(eval_msg + "\n")
                log_file.flush()

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    save_checkpoint(model, update, avg_reward, 'checkpoints/best_model', config)
                    no_improvement_count = 0
                    print(f"New best model saved. Reward: {best_avg_reward:.2f}")
                else:
                    no_improvement_count += 1

                if avg_reward >= config.target_reward:
                    print(f"Target reward {config.target_reward} achieved. Stopping training.")
                    break
                if no_improvement_count >= config.early_stopping_patience:
                    print(f"No improvement for {config.early_stopping_patience} evaluations. Early stopping.")
                    break

            if update % config.save_interval == 0:
                save_checkpoint(model, update, best_avg_reward, f'checkpoints/checkpoint_update_{update}', config)

    except (KeyboardInterrupt, RuntimeError) as e:
        print(f"\nTraining interrupted: {e}")
        save_checkpoint(model, update, best_avg_reward, 'checkpoints/interrupted_model', config)
        print("Model saved to checkpoints/interrupted_model.zip")

    finally:
        envs.close()
        log_file.close()

        save_checkpoint(model, update, best_avg_reward, 'checkpoints/final_model', config)

        with open('logs/training_history.json', 'w') as f:
            json.dump({
                'reward_history': reward_history,
                'eval_rewards': eval_rewards,
                'steps': steps,
                'eval_steps': eval_steps,
                'final_reward': best_avg_reward,
                'total_updates': update,
                'total_timesteps': model.num_timesteps
            }, f, indent=4)

        # Print final statistics.
        print(f"\nTraining completed.")
        print(f"Best average reward: {best_avg_reward:.2f}")
        print(f"Total updates: {update}")
        print(f"Total timesteps: {model.num_timesteps:,}")
        print(f"Final model saved to: checkpoints/final_model.zip")
        print(f"Best model saved to: checkpoints/best_model.zip")
        print(f"Training history saved to: logs/training_history.json")
        print(f"Use 'python visualization.py' to visualize results.")


if __name__ == "__main__":
    main()
