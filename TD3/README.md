# Twin Delayed DDPG (TD3)

## Overview
TD3 is an off-policy actor-critic algorithm for continuous action spaces that improves upon DDPG with twin Q-networks, delayed policy updates, and target policy smoothing. This implementation follows the same project layout as the other agents in this repository (PPO, SAC, DQN, etc.).

## Highlights
- Deterministic policy network with twin critics and Polyak-averaged targets.
- Configurable policy delay, exploration noise, and target smoothing noise.
- Replay buffer and warm-up phase shared with the other off-policy agents.
- Built-in WandB logging, tqdm-aware logging utilities, and checkpoint management.

## Quickstart
    python -m TD3.main train --config TD3/configs/pendulum.yaml
Provide <code>--wandb_key YOUR_KEY</code> to authenticate for logging. Checkpoints live in <code>TD3/checkpoints</code> and the moving-average best checkpoint is written to <code>best.pt</code>.

Watch a trained policy:
    python -m TD3.main demo --config TD3/configs/pendulum.yaml --model_path TD3/checkpoints/best.pt

## Configuration
YAML files under <code>TD3/configs/</code> expose the hyper-parameters:
- <strong>Environment</strong>: Gym id, render mode, and optional keyword arguments.
- <strong>Training</strong>: interaction steps, warm-up horizon, batch size, replay capacity, learning rates, Polyak coefficient, target noise, noise clip, policy delay, and exploration noise.
- <strong>Model</strong>: shared hidden layer sizes and activation for actor and critics.
- <strong>Logging</strong>: intervals, checkpoint cadence, output paths, and logger behaviour.
- <strong>Inference</strong>: default checkpoint and number of evaluation episodes.

Clone the provided <code>pendulum.yaml</code> to target other continuous-control tasks.

## References
- Fujimoto et al., Addressing Function Approximation Error in Actor-Critic Methods, ICML 2018.
- OpenAI Spinning Up TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
- Stable-Baselines3 TD3: https://stable-baselines3.readthedocs.io/en/master/modules/td3.html
