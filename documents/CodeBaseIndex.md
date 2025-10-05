# 2D Cart Pole PPO Project's Codebase Index:

## Project Overview:

This project uses the **Proximal Policy Optimization (PPO)** algorithm for the `CartPole-v1` environment from Gymnasium. It uses the `stable-baselines3` library for the core PPO algorithm and provides programs for training, evaluation, hyperparameter optimization, and visualization.

### Project Structure:
```
2DCartPolePPO/
├── Core Implementation Files
├── Training and Evaluation
├── Hyperparameter Optimization
├── Visualization and Analysis
├── Configuration and Results
├── Model Checkpoints
└── Logs and Data
```

---

## Core Implementation Files:

### 1. `config.py`:
**Purpose**: Centralized configuration for the PPO algorithm and environment.
- **Class**: `Config`
- **Key Parameters**:
  - Environment: `CartPole-v1`
  - Training timesteps: 200,000
  - PPO hyperparameters (learning rate, batch size, etc.)
  - Network architecture: (128, 128) hidden dimensions
- **Methods**:
  - `get_default_config()`: Returns a `Config` object, updated with `best_hyperparameters.json` if it exists.

### 2. `train.py`:
**Purpose**: Main training loop and environment interaction.
- **Features**:
  - Vectorized environments (8 parallel environments).
  - Automatic device detection (CPU is preferred for this task).
  - Early stopping when the target reward is reached.
  - Periodic evaluation and checkpointing.
- **Training Loop**:
  - Utilizes `stable-baselines3` PPO `learn` method.
  - Saves the best model based on evaluation performance.
- **Outputs**:
  - Model checkpoints in `checkpoints/`.
  - Training logs in `logs/`.

### 3. `evaluate.py`:
**Purpose**: Detailed agent evaluation and analysis.
- **Evaluation Metrics**:
  - Average reward and standard deviation.
  - Episode length statistics.
  - Success rate (reward ≥ 475).
- **Features**:
  - 100 episode evaluation by default.
  - Optional rendering for visualization.
  - Generates and saves detailed performance plots.
- **Outputs**:
  - `evaluation_results.json` with summary statistics.
  - `evaluation_plots.png` and `evaluation_analysis.png` with visualizations.

### 4. `hyperparameter_optimization.py`:
**Purpose**: Automated hyperparameter tuning using Optuna.
- **Optimization Framework**: Tree-structured Parzen Estimator (TPE).
- **Search Space**:
  - Network architecture.
  - PPO specific parameters (learning rate, gamma, clip range, etc.).
- **Features**:
  - Trial based model training and evaluation.
  - Pruning for early termination of unpromising trials.
- **Output**: `best_hyperparameters.json` with the optimized parameters.

### 5. `visualization.py`:
**Purpose**: Detailed training progress visualization.
- **Class**: `PPOTrainingVisualizer`
- **Data Sources**:
  - `logs/training_history.json`.
- **Visualizations**:
  - Training reward progression.
  - Evaluation performance over time.
  - Reward distribution and analysis.
- **Outputs**:
  - `ppo_training_progress.png`
  - `ppo_reward_analysis.png`

---

## Documentation and Results:

### 1. `README.md`:
**Purpose**: Main documentation for the project.
- **Contents**:
  - Overview of the PPO algorithm.
  - Implementation details and hyperparameters.
  - Project structure and data flow.
  - Usage guide for all programs.
  - Results and analysis with embedded visuals.

### 2. `documents/CodeBaseIndex.md`:
**Purpose**: This file, providing a detailed index of the project's codebase.

### 3. `results/`:
**Purpose**: Stores the output files from the evaluation program.
- `evaluation_results.json`: JSON file with the final evaluation metrics.
- `HyperparameterTuningResults.txt`: Text file with the results from the hyperparameter optimization.

### 4. `visuals/`:
**Purpose**: Contains all the visual assets used in the documentation.
- `evaluation_analysis.png`: Plot from the evaluation program.
- `evaluation_plots.png`: Plot from the evaluation program.
- `ppo_reward_analysis.png`: Plot from the visualization program.
- `ppo_training_progress.png`: Plot from the visualization program.
- `PPOTrainingInfoOutput.png`: Screenshot of the training output.

---

## Directory Structure and Data Storage:

### Model Checkpoints (`checkpoints/`):
- `best_model.zip`: The best performing model saved during training.
- `final_model.zip`: The model at the end of the training process.

### Logs (`logs/`):
- `training_history.json`: A JSON file containing the history of rewards and other metrics during training.
- `training_log.txt`: A text file with the raw output from the training program.

---

## Key Algorithm Features:

### PPO (Proximal Policy Optimization):
The implementation relies on `stable-baselines3` and is configured with the following key aspects:
1. **Policy Update**: Clipped surrogate objective.
2. **Advantage Estimation**: Generalized Advantage Estimation (GAE).
3. **Value Function**: Standard Mean Squared Error (MSE) loss for the critic.
4. **Entropy Regularization**: To encourage exploration.

### Technical Specifications:
- **Environment**: `CartPole-v1` (4D state, 2D discrete action).
- **Framework**: PyTorch with Gymnasium and Stable-Baselines3.
- **Parallelization**: 8 vectorized environments.
- **Optimization**: Adam optimizer.

---

## Usage Instructions:

### Training a New Model:
```bash
python train.py
```

### Evaluating a Trained Model:
```bash
python evaluate.py
```

### Hyperparameter Optimization:
```bash
python hyperparameter_optimization.py
```

### Visualization:
```bash
python visualization.py
```
