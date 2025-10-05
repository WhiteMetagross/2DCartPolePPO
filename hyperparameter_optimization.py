#This program optimizes the hyperparameters for the PPO agent using Optuna.
#It uses the PPO algorithm from Stable-Baselines3.
#It logs the optimization progress and saves the best hyperparameters.

import os
import optuna
import gymnasium as gym
import torch
import numpy as np
import json
from tqdm import tqdm

#Stable-Baselines3 PPO:
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


#This function evaluates the agent's performance.
def evaluate_model(model, env_name, num_episodes=5):
    eval_env = Monitor(gym.make(env_name))
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=num_episodes, deterministic=False)
    eval_env.close()
    return mean_reward

#This function saves the model checkpoint.
def save_checkpoint(model, trial_number, update, best_reward, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"trial_{trial_number}_update_{update}")
    model.save(checkpoint_path)
    return checkpoint_path

#This function is the objective function for Optuna.
def objective(trial):
    env_name = "CartPole-v1"
    seed = 17
    num_envs = 8
    device = 'cpu'

    total_timesteps = 100_000

    #Suggest hyperparameters ranges.
    net_arch_label = trial.suggest_categorical('net_arch', ['64-64', '128-128', '256-256'])
    net_arch_map = {'64-64': (64, 64), '128-128': (128, 128), '256-256': (256, 256)}
    net_arch_choice = net_arch_map[net_arch_label]
    n_steps = trial.suggest_categorical('n_steps', [128, 256, 512, 1024])
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    n_epochs = trial.suggest_int('n_epochs', 5, 15)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.98, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.98)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.02)
    vf_coef = trial.suggest_float('vf_coef', 0.3, 0.8)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 1.0)

    #Ensure batch_size divides n_steps * n_envs
    rollout_size = n_steps * num_envs
    if rollout_size % batch_size != 0:
        valid_batch_sizes = [b for b in [64, 128, 256, 512] if rollout_size % b == 0]
        batch_size = valid_batch_sizes[0] if valid_batch_sizes else max(64, min(256, rollout_size))

    envs = make_vec_env(env_name, n_envs=num_envs, seed=seed)

    #Create PPO model.
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=dict(net_arch=net_arch_choice),
        verbose=0,
        seed=seed,
        device=device
    )

    #Train and evaluate.
    try:
        model.learn(total_timesteps=total_timesteps, reset_num_timesteps=True, progress_bar=False)
        mean_reward = evaluate_model(model, env_name, num_episodes=10)
        trial.report(mean_reward, step=1)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    finally:
        envs.close()

    checkpoint_dir = f"optuna_checkpoints/trial_{trial.number}"
    save_checkpoint(model, trial.number, update=0, best_reward=mean_reward, checkpoint_dir=checkpoint_dir)

    return mean_reward

#This is the main function.
def main():
    print("Starting hyperparameter optimization for PPO (Stable-Baselines3).")
    print("This may take several minutes to hours depending on the number of trials...")

    #Create study.
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    )

    #Run optimization.
    n_trials = 50  #Number of trials for production optimization, and adjust based on available resources.
    study.optimize(objective, n_trials=n_trials)

    #Save results.
    print("\nOptimization completed.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.2f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    #Save best hyperparameters.
    with open('best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)

    print("\nBest hyperparameters saved to best_hyperparameters.json")
    print("You can now use these hyperparameters for training with train.py")


if __name__ == "__main__":
    main()
