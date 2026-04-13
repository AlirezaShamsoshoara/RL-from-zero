<p align="center">
  <img src="assets/a3c_logo.svg" alt="A3C Logo" width="520" />
</p>

# A3C (Asynchronous Advantage Actor-Critic)

## Overview
A3C is an on-policy actor-critic algorithm that achieves stable, efficient training by running multiple workers in parallel, each interacting with its own copy of the environment and asynchronously pushing gradients to a shared global network. This removes the need for experience replay and decorrelates data through parallelism rather than a replay buffer.

## What is A3C?

A3C launches multiple environment workers that run in parallel, each maintaining a local copy of the policy/value network. After every $t_{\max}$ steps (or episode termination) each worker computes gradients on its local trajectory and applies them to the shared global network, enabling on-policy updates without heavy experience replay.

Below is a compact math view aligned with this repository's implementation (see `A3C/a3c/agent.py`, `A3C/a3c/worker.py`) and written to render in both VSCode Markdown preview and GitHub.

**Actor-Critic architecture**

The shared network $f_\theta$ outputs both a policy (actor) and a value estimate (critic) from a common feature backbone:

$$
\pi_\theta(a \mid s) = \text{softmax}\!\left(W_\pi \cdot h_\theta(s) + b_\pi\right)
$$

$$
V_\theta(s) = W_v \cdot h_\theta(s) + b_v
$$

where $h_\theta(s)$ is the shared feature representation (MLP backbone).

**N-step return**

Each worker collects a trajectory of up to $t_{\max}$ steps. The n-step return for step $t$ within the rollout is:

$$
R_t = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V_\theta(s_{t+n}) \cdot (1 - d_{t+n})
$$

where $d_{t+n} \in \{0,1\}$ is the terminal indicator. In this implementation, returns are computed backwards via:

$$
R_t = r_t + \gamma (1 - d_t) \cdot R_{t+1}, \quad R_{T} = V_\theta(s_T)
$$

**Advantage**

The advantage estimate for each step is:

$$
A_t = R_t - V_\theta(s_t)
$$

**Policy loss (actor)**

The policy gradient loss uses the log-probability weighted by the advantage (with stop-gradient on $A_t$):

$$
\mathcal{L}_\pi = -\mathbb{E}_t\!\left[\log \pi_\theta(a_t \mid s_t) \cdot A_t\right]
$$

**Value loss (critic)**

The value function is trained to predict the n-step return:

$$
\mathcal{L}_V = \mathbb{E}_t\!\left[\left(V_\theta(s_t) - R_t\right)^2\right]
$$

**Entropy bonus**

An entropy bonus encourages exploration by preventing premature policy collapse:

$$
\mathcal{H}(\pi_\theta(\cdot \mid s_t)) = -\sum_a \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t)
$$

**Total loss**

The combined loss minimized by each worker is:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_\pi + c_v \mathcal{L}_V - c_e \mathbb{E}_t\!\left[\mathcal{H}(\pi_\theta(\cdot \mid s_t))\right]
$$

where $c_v$ is the value loss coefficient (`value_loss_coef`) and $c_e$ is the entropy coefficient (`entropy_coef`).

**Asynchronous gradient update**

Each worker $k$ maintains a local copy $\theta'_k$ of the global parameters $\theta$. After each rollout:

1. Sync local parameters: $\theta'_k \leftarrow \theta$
2. Collect trajectory $\{(s_t, a_t, r_t)\}_{t=0}^{n-1}$ using $\pi_{\theta'_k}$
3. Compute $\nabla_{\theta'_k} \mathcal{L}_{\text{total}}$ on local parameters
4. Apply gradients to the global network:

$$
\theta \leftarrow \theta - \alpha \cdot \nabla_{\theta'_k} \mathcal{L}_{\text{total}}
$$

with gradient clipping $\|\nabla\| \leq g_{\max}$ (`max_grad_norm`) before the update. The optimizer is a `SharedAdam` whose state tensors are placed in shared memory so all workers apply updates to the same momentum buffers.

**Parameter/term guide**
- $s_t$, $a_t$, $r_t$: state, action, reward at time $t$; $s_{t+1}$ is the next state.
- $\gamma$: discount factor; $\alpha$: learning rate.
- $\theta$: global (shared) network parameters; $\theta'_k$: local copy for worker $k$.
- $V_\theta(s)$: value function estimate; $\pi_\theta(a \mid s)$: policy (categorical distribution).
- $c_v$: value loss coefficient; $c_e$: entropy bonus coefficient.
- $g_{\max}$: gradient clipping max norm; $t_{\max}$: max rollout length per worker.
- $d_t \in \{0,1\}$: terminal indicator (1 if terminal, 0 otherwise).

## Highlights
- Shared-memory global model with `SharedAdam` optimizer across spawned worker processes via `torch.multiprocessing`.
- N-step returns computed per rollout; no replay buffer needed (on-policy).
- **Multi-environment support**: each worker can run multiple environments in parallel via `SyncVectorEnv` (`num_envs` config), multiplying data throughput.
- Configurable number of parallel workers, rollout length, and entropy regularization.
- WandB integration, tqdm-aware logging, and checkpoint management consistent with other algorithms.

## Default Environment: Acrobot-v1

Acrobot is a classic control environment where a two-link pendulum must swing its free end above a target height:
- **Observation**: 6-dimensional continuous state (joint angles and angular velocities).
- **Actions**: 3 discrete torques applied to the actuated joint ($\{-1, 0, +1\}$).
- **Reward**: $-1$ per step until the free end crosses the target height; the goal is to solve it in as few steps as possible.
- **Max episode length**: 500 steps (truncated if not solved).

This environment tests A3C's ability to learn from sparse negative reward signals using parallel exploration across workers.

## Quickstart
```bash
python -m A3C.main train --config A3C/configs/acrobot.yaml

python -m A3C.main demo --config A3C/configs/acrobot.yaml --model_path A3C/checkpoints/best.pt
```
Use `--wandb_key YOUR_KEY` to authenticate for logging, or set `WANDB_API_KEY` in your environment. Checkpoints live in `A3C/checkpoints`, and the moving-average best checkpoint is written to `best.pt`.

Setup with uv (activate the venv if you want to call `python` directly; `uv run` does not require activation):

Windows cmd:
```cmd
uv venv .venv
uv sync
.\.venv\Scripts\activate.bat
python -m A3C.main train --config A3C/configs/acrobot.yaml
python -m A3C.main demo --config A3C/configs/acrobot.yaml --model_path A3C/checkpoints/best.pt --episodes 5
```

macOS/Linux (bash or zsh):
```bash
uv venv .venv
uv sync
source .venv/bin/activate
python -m A3C.main train --config A3C/configs/acrobot.yaml
python -m A3C.main demo --config A3C/configs/acrobot.yaml --model_path A3C/checkpoints/best.pt --episodes 5
```

If you prefer `uv run` instead of activation:
```bash
uv run -m A3C.main train --config A3C/configs/acrobot.yaml
uv run -m A3C.main demo --config A3C/configs/acrobot.yaml --model_path A3C/checkpoints/best.pt --episodes 5
```

## Configuration
YAML files under `A3C/configs/` expose the hyper-parameters:
- **Environment**: Gym id, render mode.
- **Workers**: number of parallel processes and max rollout length per update.
- **Training**: total interaction steps, discount factor, entropy coefficient, value loss coefficient, learning rate, and gradient clipping.
- **Model**: shared hidden layer sizes and activation for the actor-critic backbone.
- **Logging**: intervals, checkpoint cadence, output paths, and logger behaviour.
- **Inference**: default checkpoint path and number of evaluation episodes.

Clone the provided config in `A3C/configs/` to target other discrete-action Gymnasium tasks (e.g., `CartPole-v1`, `LunarLander-v2`).

## Multi-Environment Support
A3C supports parallel environment execution within each worker via Gymnasium's `SyncVectorEnv`. The `num_envs` parameter controls how many environment instances each worker runs simultaneously. The total number of active environments is `num_workers * num_envs`.

```yaml
num_workers: 4   # OS-level processes (async gradient pushes)
num_envs: 4      # environments per worker (vectorized stepping)
# Total: 4 * 4 = 16 environments running concurrently
```

When `num_envs > 1`, each worker step collects transitions from all its environments in a single batched forward pass, then flattens the data for the gradient update. This multiplies data throughput without spawning additional processes.

> **Important:** Each parallel environment runs within the worker process. Since A3C workers are already separate OS processes, keep `num_workers * num_envs` **at or below the number of CPU cores** available on your machine. Check your core count with `nproc` (Linux) or `sysctl -n hw.ncpu` (macOS) and set values accordingly.

## Architecture

```
                           +-------------------+
                           |  Global Network   |
                           | (shared memory)   |
                           |  theta, SharedAdam|
                           +--------+----------+
                                    |
                  +-----------------+-----------------+
                  |                 |                 |
           +------v------+  +------v------+  +------v------+
           |  Worker 0   |  |  Worker 1   |  |  Worker k   |
           | local theta |  | local theta |  | local theta |
           | env 0..E-1  |  | env 0..E-1  |  | env 0..E-1  |
           | (VectorEnv) |  | (VectorEnv) |  | (VectorEnv) |
           +------+------+  +------+------+  +------+------+
                  |                 |                 |
             grads + episodes  grads + episodes  grads + episodes
                  |                 |                 |
                  +-----------------+-----------------+
                                    |
                           +--------v----------+
                           |  Result Queue     |
                           | (main process     |
                           |  logging/ckpt)    |
                           +-------------------+
        E = num_envs per worker; total envs = num_workers * num_envs
```

## Comparison with Other Policy Gradient Methods

| Aspect | A3C | PPO | SAC |
|--------|-----|-----|-----|
| **Update style** | On-policy, async | On-policy, sync | Off-policy |
| **Parallelism** | Multiple processes | Vectorized envs | Single process + replay |
| **Experience reuse** | No replay buffer | No replay buffer | Replay buffer |
| **Action space** | Discrete | Discrete | Continuous |
| **Exploration** | Entropy bonus | Entropy bonus | Maximum entropy framework |
| **Gradient sync** | Async (lock-free) | Synchronous | N/A |

## Notes
- This reference implementation currently supports CPU devices, flat observation spaces, and discrete action spaces.
- Increase `num_workers` and `t_max` carefully; CPU contention can hurt performance with too many workers.
- Tweak exploration via `entropy_coef` in the config to balance return and stability.
- Unlike PPO which uses synchronized vectorized environments, A3C uses true multiprocessing with independent environment instances per worker.

## References

- **Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harber, T., Silver, D., & Kavukcuoglu, K. (2016).** *Asynchronous Methods for Deep Reinforcement Learning.* Proceedings of the 33rd International Conference on Machine Learning (ICML), 48, 1928-1937.
  - Original A3C paper: https://arxiv.org/abs/1602.01783

- **Sutton, R. S. & Barto, A. G. (2018).** *Reinforcement Learning: An Introduction (2nd ed.).* MIT Press.
  - Chapter 13 covers policy gradient methods; Section 13.5 discusses actor-critic architectures.

- **Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2016).** *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR 2016.
  - GAE paper relevant to advantage estimation: https://arxiv.org/abs/1506.02438

- **OpenAI Spinning Up - Key Papers in Deep RL:**
  - A3C entry and related algorithms: https://spinningup.openai.com/en/latest/spinningup/keypapers.html

- **Arthur Juliani (2016).** *Simple Reinforcement Learning with Tensorflow Part 8: Asynchronous Actor-Critic Agents (A3C).*
  - Accessible tutorial walkthrough: https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2
