#This program provides a centralized configuration for the CartPole-v1 environment and the PPO algorithm.
#It can be imported by training, evaluation, and optimization scripts.

from dataclasses import dataclass, asdict
from typing import Dict, Any
import os
import json

#This class defines the configuration for the CartPole-v1 environment and the PPO algorithm.
@dataclass
class Config:
    #Environment
    env_name: str = 'CartPole-v1'
    seed: int = 17
    num_envs: int = 8

    #Training budget
    total_timesteps: int = 200_000

    #PPO hyperparameters (SB3 names)
    n_steps: int = 1024          
    batch_size: int = 256
    n_epochs: int = 10           
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    #Policy network
    policy_hidden_dims: tuple = (128, 128)

    #Logging / evaluation
    eval_interval: int = 2       
    save_interval: int = 20       
    log_interval: int = 5         
    target_reward: float = 500.0
    early_stopping_patience: int = 20

    def get_dict(self) -> Dict[str, Any]:
        return asdict(self)

#This function returns the default configuration for the CartPole-v1 environment and the PPO algorithm.
def get_default_config() -> Config:
    cfg = Config()

    #If best_hyperparameters.json exists, merge its values into the default config.
    path = os.path.join(os.getcwd(), 'best_hyperparameters.json')
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                best = json.load(f)

            #Map known PPO hyperparameters from file to config fields.
            for key in ['n_steps', 'batch_size', 'n_epochs', 'learning_rate', 'gamma',
                        'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm']:
                if key in best:
                    setattr(cfg, key, best[key])

            if 'net_arch' in best:
                net_arch = best['net_arch']
                if isinstance(net_arch, str):
                    try:
                        dims = tuple(int(x) for x in net_arch.split('-') if x)
                        if len(dims) > 0:
                            cfg.policy_hidden_dims = dims
                    except ValueError:
                        pass
                elif isinstance(net_arch, (list, tuple)):
                    try:
                        cfg.policy_hidden_dims = tuple(int(x) for x in net_arch)
                    except Exception:
                        pass
        except Exception:
            pass

    return cfg
