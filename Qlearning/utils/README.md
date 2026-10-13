# Q-learning evaluation and assets

Optional reproducibility helpers for the **tabular** Q-learning agent. These
helpers do not introduce neural networks, replay buffers, or actor/critic losses.
Run from the repository root; wrappers change to the root themselves.

## Fresh uv environment

Choose an unused, custom environment name for each validation run. For example:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venvs/qlearning-heldout"
uv venv "$UV_PROJECT_ENVIRONMENT"
uv sync
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

Dependencies are existing project packages: Gymnasium, NumPy, PyTorch (checkpoint
serialization only), W&B, Matplotlib, imageio, and Pillow. FrozenLake uses toy-text
and pygame rendering, not Box2D. The agent and evaluation remain CPU-only.
`uv run` below uses the selected `UV_PROJECT_ENVIRONMENT`; wrappers respect
`PYTHON`, defaulting to the activated environment's `python`. Only load trusted
checkpoints through the core PyTorch checkpoint loader.

## Unfiltered checkpoint evaluation

```bash
uv run python -m Qlearning.utils.evaluate \
  --config Qlearning/configs/frozenlake.yaml \
  --episodes 100 --seed 10000 --out Qlearning/logs/evaluation.json

# Seeded random-policy control, on the same config and seed range
uv run python -m Qlearning.utils.evaluate \
  --config Qlearning/configs/frozenlake.yaml --random \
  --episodes 100 --seed 10000 --out Qlearning/logs/random_evaluation.json

# Wrappers use the activated Python
bash Qlearning/utils/evaluate.sh --episodes 100 --seed 10000
bash Qlearning/utils/evaluate.sh --random --out Qlearning/logs/random_evaluation.json
```

CLI defaults: `--config Qlearning/configs/frozenlake.yaml`, `--episodes 100`,
`--seed 10000`, and `--out Qlearning/logs/evaluation.json`. `--checkpoint PATH` is
optional: when absent, the checkpoint is read from **that config's**
`inference_model_path`, avoiding mismatches when selecting another config.
`--random` does not load a checkpoint.

The loaded Q table must match the environment's zero-based `Discrete` state and
action spaces and contain finite values. The deterministic policy takes the
**first argmax** action, with no epsilon exploration, random tie-breaking,
learning, or mutation of the agent's step counter. Episode `i` resets with
`seed + i` and independently seeds the action space. Every episode stops on
termination, truncation, or `cfg.max_steps_per_episode`.

The JSON contains every return, length, seed, discounted return, initial state,
max-action start Q, and stop reason, plus count/mean/population std/min/max,
configuration, checkpoint episode/step metadata, SHA-256, and the local W&B run
location. For the random control, start values and checkpoint metadata are null.
All individual episodes and aggregate metrics are logged to **forced offline
W&B**, even if the inherited mode is online or disabled. No login or cloud sync
is performed.

- Undiscounted return: sum of all episode rewards.
- Discounted return: `sum_t gamma**t * reward_t`, with the first reward at `t=0`.
- FrozenLake success: fraction of **all evaluated episodes** with total return
  greater than zero. The tool reports this definition explicitly and does not
  apply it to arbitrary non-FrozenLake tasks.
- Start value: `max_a Q(initial_state, a)`, not a neural-network prediction.

### What fresh seeds do and do not establish

Training resets episode `i` with `cfg.seed + i`. The default 5,000-episode
configuration therefore uses seeds 42-5041; evaluation starts at 10000, outside
that range. If changing the training seed or episode budget, select a held-out
range explicitly rather than assuming the default remains disjoint.

On a fixed, non-slippery FrozenLake map with a fixed initial state, greedy-policy
rollouts repeat the same path. A 100-episode success rate then describes
**same-map deterministic repetitions**, not 100 independent starting conditions,
stochastic robustness, or generalization to unseen maps. Random-policy seeds
change random actions, not the fixed map. A slippery config changes transition
dynamics; alternate maps change the state semantics and require corresponding
training/checkpoints. Do not relabel same-map results as generalization.

## Slow deterministic GIF

```bash
uv run python -m Qlearning.utils.make_gif \
  --config Qlearning/configs/frozenlake.yaml \
  --out Qlearning/assets/qlearning_frozenlake_8x8.gif \
  --episodes 8 --keep-top 3 --fps 6 --stride 1 --seed 3000

bash Qlearning/utils/make_gif.sh
```

The shown arguments are the defaults. Checkpoint inference follows the selected
config unless `--checkpoint` is provided. Rendering is headless with
`SDL_VIDEODRIVER=dummy` and `rgb_array`. Frames are copied **before stepping** so
later renders cannot mutate earlier frames. The final reached state is included
and held for a second. Six fps and stride one keep a short, roughly 14-step
8x8 route watchable instead of speeding past it.

The tool prints all returns and selected returns, then keeps the top
`--keep-top` episodes. **This is a selected showcase, not an unbiased success-rate
estimate.** Report metrics from `evaluate.py`, which retains every requested
episode. Positive episodes/fps/stride/keep-top are required, and keep-top must
not exceed episodes. Inspect the generated GIF before publishing.

## Tabular training charts

```bash
uv run python -m Qlearning.utils.make_charts \
  --run wandb/offline-run-EXAMPLE/run-EXAMPLE.wandb \
  --out-dir Qlearning/assets --title "Q-learning FrozenLake 8x8" \
  --summary Qlearning/logs/chart_metrics.json

RUN=wandb/offline-run-EXAMPLE/run-EXAMPLE.wandb \
  SUMMARY=Qlearning/logs/chart_metrics.json bash Qlearning/utils/make_charts.sh
```

Replace the example with the actual **Qlearning training** datastore. The
reader handles repeated `nested_key` fields, closes the datastore, and sorts
training records by `progress/episode`. It does not use the newest run implicitly,
which could belong to an evaluation or another algorithm. Chart generation reads
only local logs and uses Matplotlib's noninteractive Agg backend.

- `chart_01.png`: discounted max-start Q versus discounted epsilon-behavior
  training return, plus a separate panel for raw undiscounted episode return.
  **The policies differ:** Q targets the greedy policy, but training data contain
  epsilon exploration. Different policies and capped finite horizons mean the
  gap is **not calibration proof**, even though both top-panel series are
  discounted. Raw undiscounted returns are not matching Q targets either.
- `chart_02.png`: mean squared TD error and **mean absolute** TD error with
  distinct axes, plus epsilon. These are tabular-update diagnostics, not fake
  actor/critic or neural-network optimization losses.
- `chart_03.png`: episode return and length, plus FrozenLake success rate.

Required finite metrics: `charts/avg_return`, `charts/avg_length`,
`charts/success_rate`, `charts/epsilon`, `stats/start_q`,
`stats/discounted_return`, `stats/td_error`, and `loss/td_mse`.
`progress/steps` is included in the optional JSON if present. Missing required
metrics or missing matched Q/discounted-return rows produce a clear error,
not blank charts. Epsilon/success must be fractions in `[0,1]`, and TD diagnostics
must be nonnegative.

`--smooth` defaults to 9 **logged points**, using a centered display-only mean;
it is not a window in training episodes. `--summary PATH` optionally writes raw
metric series, counts, first/last values and episodes, means/minima/maxima, units,
and policy-comparison caveats. No smoothing is applied to summary statistics.
Inspect generated PNGs visually before publishing.

## Wrapper overrides

All wrappers accept `PYTHON` and forward extra CLI flags, which override defaults.
Evaluation/GIF accept `CONFIG`, `OUT`, and optional `CHECKPOINT`; when unset,
`CHECKPOINT` is inferred by Python from the config, not hardcoded in the wrapper.
Charts accept `RUN`, `OUT_DIR`, `TITLE`, and optional `SUMMARY`.

```bash
PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python" bash Qlearning/utils/evaluate.sh
CONFIG=/path/to/config.yaml bash Qlearning/utils/evaluate.sh --seed 2000
bash Qlearning/utils/make_charts.sh --run /path/to/run-ID.wandb --smooth 5
```
