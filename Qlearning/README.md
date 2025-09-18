<p align="center">
  <img src="assets/qlearning-logo.svg" width="420" alt="Q-Learning Logo" />
</p>

# Q-learning quickstart

## What is Q-learning?
- Q-learning is a model-free, off-policy, value-based RL algorithm that learns an action-value (Q) function for discrete state and action spaces via temporal-difference updates and an epsilon-greedy behavior policy.

This template mirrors the PPO structure and includes:
- Clean Config class backed by a YAML file in `Qlearning/configs/`
- Fire CLI with two commands: `train` and `demo`
- tqdm progress bars
- Weights & Biases logging for rewards
- Checkpoint saving (best and periodic) storing the Q-table

## Default environment
- Uses `FrozenLake-v1` (toy-text) with `is_slippery: false` for a deterministic grid. Both observations and actions are discrete, ideal for tabular Q-learning.

## Setup with uv (Windows cmd):
1) Create venv and install deps
   uv venv .venv
   uv sync

2) Train Q-learning on FrozenLake
   uv run -m Qlearning.main train --config Qlearning/configs/frozenlake.yaml

3) Demo a trained agent (renders in terminal)
   uv run -m Qlearning.main demo --config Qlearning/configs/frozenlake.yaml --model_path Qlearning/checkpoints/best.pt --episodes 5

Notes
- Only discrete observation and action spaces are supported.
- You can edit hyperparameters in `Qlearning/configs/frozenlake.yaml`.


# References and useful links - Papers:
- [Technical Note of Q-learning](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)
- [Reinforcement Learning: An Introduction; Richard S. Sutton and Andrew G. Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- [Reinforcement Learning An Introduction second edition; Richard S. Sutton and Andrew G. Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
- [Reinforcement Learning in 60 days](https://github.com/andri27-ts/Reinforcement-Learning)