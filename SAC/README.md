![Soft Actor-Critic Logo](assets/sac_logo.svg)

# Soft Actor-Critic (SAC)

## What is SAC?
- Soft Actor-Critic is an off-policy actor-critic algorithm for continuous action spaces that optimizes expected reward while maximizing policy entropy, leveraging stochastic policies, twin Q-value critics, and automatic temperature tuning to balance exploration and stability.

Below is a compact math view aligned with this repository's implementation (see `SAC/sac/agent.py`) and written to render in both VSCode Markdown preview and GitHub.

**Maximum-entropy objective**
$$
J(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t \left(r_t + \alpha \mathcal{H}(\pi(\cdot \mid s_t))\right)\right]
$$
$$
\mathcal{H}(\pi(\cdot \mid s)) = -\mathbb{E}_{a \sim \pi}\left[\log \pi(a \mid s)\right]
$$

**Policy (squashed Gaussian with action scaling)**
$$
z = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon,\quad \epsilon \sim \mathcal{N}(0, I)
$$
$$
u = \tanh(z),\quad a = u \odot c + b
$$

**Log-probability with tanh correction**
$$
\log \pi_\theta(a \mid s) = \sum_i \left[\log \mathcal{N}(z_i; \mu_i, \sigma_i) - \log c_i - \log(1 - \tanh(z_i)^2 + \varepsilon)\right]
$$

**Critic target and loss (twin Q)**
$$
y_t = r_t + \gamma (1 - d_t)\left(\min_{i=1,2} Q_{\bar{\phi}_i}(s_{t+1}, a_{t+1}) - \alpha \log \pi_\theta(a_{t+1} \mid s_{t+1})\right)
$$
$$
L_Q = \mathbb{E}_{(s,a,r,s',d)\sim \mathcal{D}}\left[(Q_{\phi_1}(s,a) - y_t)^2 + (Q_{\phi_2}(s,a) - y_t)^2\right]
$$

**Actor loss**
$$
L_\pi = \mathbb{E}_{s\sim \mathcal{D},\, a\sim \pi_\theta}\left[\alpha \log \pi_\theta(a \mid s) - \min_{i=1,2} Q_{\phi_i}(s,a)\right]
$$

**Temperature (entropy) loss**
$$
L_\alpha = \mathbb{E}_{a\sim \pi_\theta}\left[-\log \alpha \left(\log \pi_\theta(a \mid s) + \mathcal{H}_{\text{target}}\right)\right]
$$
$$
\mathcal{H}_{\text{target}} = -|A| \cdot \text{target\_entropy\_scale}
$$

**Target-network update (Polyak averaging)**
$$
\bar{\phi} \leftarrow \tau \phi + (1 - \tau)\bar{\phi}
$$

**Parameter/term guide**
- $s_t$, $a_t$, $r_t$: state, action, reward at time $t$; $s_{t+1}$ is the next state.
- $\gamma$: discount factor; $\tau$: target network smoothing coefficient.
- $\theta$: actor parameters; $\phi_1,\phi_2$: critic parameters; $\bar{\phi}_1,\bar{\phi}_2$: target critic parameters.
- $\alpha$: temperature coefficient controlling entropy strength; $|A|$ is action dimension.
- $\mu_\theta(s)$, $\sigma_\theta(s)$: actor outputs (mean and std); $\odot$ is elementwise multiply.
- $c = (a_{\max} - a_{\min})/2$, $b = (a_{\max} + a_{\min})/2$ are action scale/bias from env bounds.
- $d_t \in \{0,1\}$: terminal indicator (1 if terminal, 0 otherwise); $\mathcal{D}$: replay buffer.
- $\varepsilon$: small constant for numerical stability in the log-prob correction.

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
- **Environment**: id, render mode, and optional kwargs passed to `gymnasium.make`.
- **Training**: total interaction steps, warmup steps, batch size, buffer size, Polyak coefficient `tau`, learning rates, and entropy target scaling.
- **Model**: hidden layer sizes and activation function shared by actor and critics.
- **Logging**: intervals, checkpointing behaviour, and logger destinations.
- **Inference**: default checkpoint for demos and number of rollout episodes.

Feel free to duplicate the YAML stub to experiment with other continuous Gym tasks.


# References, useful links, and Papers:

- Original SAC paper: https://arxiv.org/abs/1801.01290
- SAC implementation in PyTorch: https://github.com/pranz24/pytorch-soft-actor-critic
- SAC implementation in JAX: https://github.com/haarnoja/sac_jax
