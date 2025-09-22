# Deep Q-Network (DQN)

## What is DQN?
Deep Q-Networks combine Q-learning with deep neural networks to approximate state-action values in environments with high-dimensional observations. This implementation mirrors the repository structure used for PPO, A3C, SAC, and tabular Q-learning, delivering an off-policy value-based agent for discrete-action Gymnasium tasks such as `CartPole-v1`.

## Features
- Torch-based Q-network with target network syncs and optional Double DQN updates.
- Replay buffer, warmup period, and configurable exploration schedule.
- Weights & Biases logging, tqdm-compatible logging helpers, and checkpoint handling aligned with the other algorithms.
- YAML-driven hyperparameters with `Config.from_yaml` parity across the repository.

## Quickstart
```bash
python -m deepQN.main train --config deepQN/configs/cartpole.yaml
```
Provide `--wandb_key YOUR_KEY` to authenticate for cloud logging. Checkpoints land in `deepQN/checkpoints`, and the best moving-average model is saved as `best.pt`.

To run evaluation rollouts:
```bash
python -m deepQN.main demo --config deepQN/configs/cartpole.yaml --model_path deepQN/checkpoints/best.pt
```

## Configuration
All tunables live under `deepQN/configs/`. The default `cartpole.yaml` exposes:
- **Environment**: Gym id, seed, render mode, and optional kwargs for `gym.make`.
- **Model**: hidden layer sizes and activation shared by online and target Q networks.
- **Training**: interaction horizon, batch size, replay capacity, learning rate, target-sync cadence, and gradient clipping.
- **Exploration**: epsilon schedule endpoints and decay horizon plus evaluation epsilon.
- **Logging**: intervals, checkpoint policy, and logger sinks.
- **Inference**: default checkpoint path and number of evaluation episodes.

Clone the YAML template to configure other discrete-control environments.

## References
- Mnih et al., "Human-level control through deep reinforcement learning" (2015)
- OpenAI Baselines DQN: https://github.com/openai/baselines/tree/master/baselines/deepq
- Stable-Baselines3 DQN: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
