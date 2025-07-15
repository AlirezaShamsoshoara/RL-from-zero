<p align="center">
  <img src="assets/iql_logo.svg" width="420" alt="Implicit Q-Learning Logo" />
</p>

<h1 align="center">Implicit Q-Learning (IQL)</h1>

## Overview
Implicit Q-Learning is an offline reinforcement learning algorithm that learns a value function with expectile regression, trains critics against the learned value target, and fits a policy via advantage-weighted behaviour cloning. This implementation follows the shared layout used across the repository (PPO, SAC, TD3, etc.) with modular configs, reusable utilities, tqdm-aware logging, and Weights & Biases instrumentation.

Key building blocks:
- Twin Q critics with soft value targets and configurable expectile, temperature, and weight clipping.
- Offline dataset loader supporting on-the-fly random rollouts, custom `.npz` buffers, or D4RL datasets.
- Weighted behaviour cloning actor that respects environment action bounds through a tanh-squashed Gaussian policy.
- Checkpoint helpers and demo entrypoints consistent with existing agents.

## Quickstart
```bash
# Train with a randomly collected Pendulum dataset
python -m IQL.main train --config IQL/configs/pendulum_random.yaml

# Evaluate a trained policy (renders by default)
python -m IQL.main demo --config IQL/configs/pendulum_random.yaml --model_path IQL/checkpoints/best.pt
```
Pass `--wandb_key YOUR_KEY` to the `train` command if you want the script to authenticate with W&B automatically. Checkpoints and best-model snapshots are written under `IQL/checkpoints/`.

## Datasets
The dataset subsystem supports three sources:
1. `random` (default) — collects `dataset_steps` transitions with a random policy in `dataset_env_id` (falls back to `env_id`). This keeps the example runnable without external files.
2. `npz` — loads offline data from a NumPy archive containing `observations`, `actions`, `rewards`, `next_observations`, and `terminals`/`dones` (+ optional `timeouts`).
3. `d4rl` — pulls datasets via `env.get_dataset()`; requires the `d4rl` package to be installed.

Reward scaling, shifting, and normalisation are available in the config to match common offline RL preprocessing schemes.

## Configuration
YAML files in `IQL/configs/` expose the main hyper-parameters:
- **Environment**: `env_id`, render mode, and kwargs forwarded to Gym.
- **Dataset**: source selector, optional alternate environment id/path, collection horizon, and reward transforms.
- **Training**: total gradient updates, batch size, expectile, temperature (`beta`), weight clipping, and optimiser settings.
- **Model**: shared hidden layer widths and activation for actor, critics, and value network.
- **Logging & Checkpoints**: logging cadence, evaluation cadence, checkpoint directory, and verbosity options.
- **Inference**: default checkpoint path and demo episode count.

Copy `pendulum_random.yaml` and tweak the fields for other continuous control tasks or offline datasets.

## References
- Kostrikov et al., Offline Reinforcement Learning with Implicit Q-Learning, NeurIPS 2021. https://arxiv.org/abs/2110.06169
- Ilya Kostrikov’s IQL implementation: https://github.com/ikostrikov/implicit_q_learning
- D4RL benchmark suite: Fu et al., D4RL: Datasets for Deep Data-Driven Reinforcement Learning, NeurIPS 2020. https://arxiv.org/abs/2004.07219
- Implicit Q-learning by the Paper Author (Ilya Kostrikov): https://github.com/ikostrikov/implicit_q_learning
- Sergey Levine Video on (Offline Reinforcement Learning: BayLearn 2021 Keynote Talk) https://www.youtube.com/watch?v=k08N5a0gG0A
