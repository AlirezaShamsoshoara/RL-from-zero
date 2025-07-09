<p align="center">
  <img src="assets/trpo_logo.svg" alt="TRPO — Trust Region Policy Optimization" width="520" />
</p>

# TRPO (Trust Region Policy Optimization) quickstart

## What is TRPO?
- TRPO is a second-order policy-gradient method that constrains each update within a trust region by solving a constrained optimization problem with a KL divergence bound. It uses conjugate gradient to approximate the natural policy gradient and a line search to guarantee monotonic policy improvement. This template follows the canonical continuous-control setup with GAE advantages and a separate value baseline.

This template includes:
- Config-driven hyperparameters via `Config` + YAML in `TRPO/configs/`
- Fire CLI with `train` and `demo` commands
- Vectorized Gym environments, tqdm progress, and WandB logging
- Conjugate-gradient solver, line search, and KL diagnostics
- Checkpoint utilities plus deterministic evaluation mode

## Quick Commands
```bash
python -m TRPO.main train --config TRPO/configs/pendulum.yaml
python -m TRPO.main demo --config TRPO/configs/pendulum.yaml --model_path TRPO/checkpoints/best.pt
```

## Configuration
YAML files in `TRPO/configs/` expose the experiment knobs:
- **Environment**: Gym id, render mode, vectorised env count, and optional `env_kwargs`.
- **Training**: total timesteps, rollout horizon, discount/GAE factors, KL threshold, conjugate-gradient settings, line-search coefficients, value-loss optimisation steps, and entropy bonus.
- **Model**: shared hidden sizes and activation for the policy/value networks.
- **Logging**: checkpoint cadence, log intervals, paths, and console/file toggles alongside WandB metadata.
- **Inference**: default checkpoint path and evaluation episode count for `demo`.

Setup with uv (Windows cmd):
1) Create venv and install deps  
   uv venv .venv  
   uv sync

2) Train TRPO on Pendulum  
   uv run -m TRPO.main train --config TRPO/configs/pendulum.yaml

3) Demo a trained policy (renders a window)  
   uv run -m TRPO.main demo --config TRPO/configs/pendulum.yaml --model_path TRPO/checkpoints/best.pt --episodes 5

## Notes
- Continuous `Box` action spaces and flat observation vectors are supported in this implementation.
- Rendering requires a local display; ensure `pygame` is installed (already in `pyproject.toml`).
- Hyperparameters can be tuned in `TRPO/configs/pendulum.yaml`. Lowering `max_kl` or increasing `cg_iters` improves stability at the cost of runtime.

## References
- Schulman et al., Trust Region Policy Optimization, ICML 2015 https://arxiv.org/abs/1502.05477
- OpenAI Spinning Up TRPO: https://spinningup.openai.com/en/latest/algorithms/trpo.html
- Stable-Baselines3 TRPO docs: https://sb3-contrib.readthedocs.io/en/master/modules/trpo.html
