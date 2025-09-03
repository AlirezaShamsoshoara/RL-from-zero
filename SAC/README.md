# Soft Actor-Critic (SAC)

This implementation mirrors the structure used for the existing PPO and Q-learning agents in this repository. It trains a Soft Actor-Critic agent on continuous-control tasks from Gymnasium, with default settings targeting `Pendulum-v1`.

## Features
- Torch-based stochastic actor with twin Q critics and target networks.
- Replay buffer with configurable capacity and warmup phase.
- Automatic entropy tuning with configurable target scale.
- WandB integration, logging utilities, and checkpointing consistent with the other algorithms.

## Quickstart
```bash
python -m SAC.main train --config SAC/configs/pendulum.yaml
```
Set `--wandb_key YOUR_KEY` if you need to authenticate programmatically. Checkpoints are written to `SAC/checkpoints` and the best moving-average model is stored as `best.pt`.

To watch the trained policy:
```bash
python -m SAC.main demo --config SAC/configs/pendulum.yaml --model_path SAC/checkpoints/best.pt
```

## Configuration
Hyper-parameters live in YAML files under `SAC/configs/`. The default file exposes:
- **Environment**: id, render mode, and optional kwargs passed to `gym.make`.
- **Training**: total interaction steps, warmup steps, batch size, buffer size, Polyak coefficient `tau`, learning rates, and entropy target scaling.
- **Model**: hidden layer sizes and activation function shared by actor and critics.
- **Logging**: intervals, checkpointing behaviour, and logger destinations.
- **Inference**: default checkpoint for demos and number of rollout episodes.

Feel free to duplicate the YAML stub to experiment with other continuous Gym tasks.
