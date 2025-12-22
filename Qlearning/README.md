<p align="center">
  <img src="assets/qlearning-logo.svg" width="420" alt="Q-Learning Logo" />
</p>

# Q-learning quickstart

<p align="center">
  <video src="assets/qlearning_frozenlake_8x8.mp4" width="420" autoplay loop muted controls></video>
</p>

## What is Q-learning?
- Q-learning is a model-free, off-policy, value-based RL algorithm that learns an action-value (Q) function for discrete state and action spaces via temporal-difference updates and an epsilon-greedy behavior policy.

## Mathematics (Q-learning & Temporal Difference)

**Bellman optimality:**

$$
Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^{*}(s', a')\right]
$$

**One-step TD target:**

$$
y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a')
$$

**TD error:**

$$
\delta_t = y_t - Q_t(s_t, a_t)
$$

**Update rule:**

$$
Q_{t+1}(s_t, a_t) = Q_t(s_t, a_t) + \alpha \, \delta_t
$$

**Epsilon-greedy behavior:**  
Take $\arg\max_a Q(s, a)$ with probability $1 - \varepsilon$; otherwise sample a random action.

This template mirrors the PPO structure and includes:
- Clean Config class backed by a YAML file in `Qlearning/configs/`
- Fire CLI with two commands: `train` and `demo`
- tqdm progress bars
- Weights & Biases logging for rewards
- Checkpoint saving (best and periodic) storing the Q-table

## Quick Commands
```bash
python -m Qlearning.main train --config Qlearning/configs/frozenlake.yaml
python -m Qlearning.main demo --config Qlearning/configs/frozenlake.yaml --model_path Qlearning/checkpoints/best.pt
# add --render True to visualize the demo in human render mode
```

## Default environment
- Uses `FrozenLake-v1` (toy-text) with `is_slippery: false` for a deterministic grid. Both observations and actions are discrete, ideal for tabular Q-learning.

## Setup with uv
### Linux/macOS (bash or zsh)
1) Create venv and install deps  
   `uv venv .venv && source .venv/bin/activate && uv sync`

2) Train Q-learning on FrozenLake  
   `uv run -m Qlearning.main train --config Qlearning/configs/frozenlake.yaml`

3) Demo a trained agent (renders in terminal)  
   `uv run -m Qlearning.main demo --config Qlearning/configs/frozenlake.yaml --model_path Qlearning/checkpoints/best.pt --episodes 5`

### Windows (cmd)
1) Create venv and install deps  
   `uv venv .venv && .\.venv\Scripts\activate && uv sync`

2) Train Q-learning on FrozenLake  
   `uv run -m Qlearning.main train --config Qlearning/configs/frozenlake.yaml`

3) Demo a trained agent (renders in terminal)  
   `uv run -m Qlearning.main demo --config Qlearning/configs/frozenlake.yaml --model_path Qlearning/checkpoints/best.pt --episodes 5`

Notes
- If `WANDB_API_KEY` is set, training will use it automatically; otherwise existing auth behavior is unchanged.
- Only discrete observation and action spaces are supported.
- You can edit hyperparameters in `Qlearning/configs/frozenlake.yaml`.
- On headless setups (e.g., WSL without a display), set `render_mode: ansi` in the config and skip `--render True` to print the demo frames to the terminal.


# References, useful links, and Papers:
- [Technical Note of Q-learning](https://www.gatsby.ucl.ac.uk/~dayan/papers/cjch.pdf)
- [Reinforcement Learning: An Introduction; Richard S. Sutton and Andrew G. Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- [Reinforcement Learning An Introduction second edition; Richard S. Sutton and Andrew G. Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
- [Reinforcement Learning in 60 days](https://github.com/andri27-ts/Reinforcement-Learning)
- [HF Q-learning example](https://huggingface.co/learn/deep-rl-course/en/unit2/q-learning-example)
- [HF Q-learning Course](https://huggingface.co/learn/deep-rl-course/unit2/introduction)
- [YTube - Q-Learning Tutorial 1: Train Gymnasium FrozenLake-v1 with Python Reinforcement Learning](https://www.youtube.com/watch?v=ZhoIgo3qqLU)
