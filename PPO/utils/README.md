# PPO reproducibility helpers

Optional helpers for fresh-seed checkpoint evaluation and README assets. Run
from the repository root. The shell wrappers change to the repo root themselves
and use `PYTHON`, defaulting to `python` from your activated environment. They do
not select or create a shared conda environment. Activate your fresh, custom-named
uv environment with the project dependencies installed before running them.

Dependencies are the existing project packages: Gymnasium, NumPy, PyTorch, W&B,
Matplotlib, imageio, and Pillow. CartPole needs no Box2D extras. Load only trusted
checkpoints because the core checkpoint loader uses PyTorch serialization.

## Deterministic evaluation

```bash
python -m PPO.utils.evaluate \
  --config PPO/configs/cartpole_tuned.yaml \
  --checkpoint PPO/checkpoints_tuned/best.pt \
  --episodes 50 --seed 1000 --out PPO/logs/evaluation.json

# Seeded random-policy control, using the same evaluation seeds
python -m PPO.utils.evaluate \
  --config PPO/configs/cartpole_tuned.yaml --random \
  --episodes 50 --seed 1000 --out PPO/logs/random_evaluation.json

# Equivalent wrapper; extra CLI flags override defaults
bash PPO/utils/evaluate.sh --episodes 50 --seed 1000
bash PPO/utils/evaluate.sh --random --out PPO/logs/random_evaluation.json
```

Flags: `--config`, `--checkpoint`, `--episodes` (50), `--seed` (1000),
`--out` (`PPO/logs/evaluation.json`), and `--random`. Random mode does not load a
checkpoint. Episode `i` resets the environment and seeds its action space with
`seed + i`. Checkpoint mode uses the deterministic categorical argmax policy on
CPU with the loaded model in eval mode. These default seeds differ from the
training config's seed; select your own held-out range if training used them.

Evaluation includes every requested episode, with no top-episode filtering. The
JSON records count, mean, population standard deviation, minimum, maximum,
individual returns, lengths, seeds, configuration, checkpoint step/best return,
checkpoint SHA-256, and the local W&B run location. Returns are **undiscounted
sums of episode rewards**. W&B is explicitly forced to `offline`, including when
the caller inherited online or disabled mode. Each episode and aggregate metrics
are logged locally, with no login or cloud synchronization.

## Demo GIF

```bash
python -m PPO.utils.make_gif \
  --config PPO/configs/cartpole_tuned.yaml \
  --checkpoint PPO/checkpoints_tuned/best.pt \
  --out PPO/assets/ppo_cartpole.gif \
  --episodes 8 --keep-top 3 --seed 3000 --fps 30 --stride 2

bash PPO/utils/make_gif.sh
```

The defaults match the example. Rendering is headless (`SDL_VIDEODRIVER=dummy`,
`rgb_array`) and actions are deterministic. The tool prints every episode's
return, then selects the top `--keep-top` episodes, subsamples with `--stride`,
and holds the last frame briefly between episodes. **The GIF is a selected
showcase, not an unbiased evaluation.** Use `evaluate.py` for reported metrics.
`--keep-top` cannot exceed `--episodes`; episodes, fps, stride, and keep-top must
be positive.

## Training charts

```bash
python -m PPO.utils.make_charts \
  --run wandb/offline-run-EXAMPLE/run-EXAMPLE.wandb \
  --out-dir PPO/assets --title "PPO CartPole-v1" \
  --summary PPO/logs/chart_metrics.json

RUN=wandb/offline-run-EXAMPLE/run-EXAMPLE.wandb \
  SUMMARY=PPO/logs/chart_metrics.json bash PPO/utils/make_charts.sh
```

Choose the actual **PPO training** datastore path explicitly. The wrapper does
not guess from the newest W&B run, which could belong to an evaluation or another
algorithm. The reader handles repeated `nested_key` fields and selects history
rows with `progress/global_step`, sorted by environment-transition count. No
network access is needed. `--smooth` defaults to 9 logged points; it is a centered
moving average for display only, not a window in environment steps.

- `chart_01.png`: two panels. The first compares `stats/value_mean` with
  `stats/return_target_mean` from **matching rollout rows**, both in discounted
  value/GAE-target units. The second shows `charts/avg_return`, an undiscounted
  episode reward sum. Those quantities have different units: do not interpret
  their difference as overestimation or claim calibration from matching means.
- `chart_02.png`: policy and value losses, with separate axes.
- `chart_03.png`: undiscounted episode return and episode length.

Required metrics are `charts/avg_return`, `charts/avg_length`, `stats/value_mean`,
`stats/return_target_mean`, `loss/policy`, and `loss/value`. Missing finite data or
missing matched value/target rows cause a clear error instead of blank charts.
The optional `--summary` JSON reports raw metric counts, first/last values and
steps, means, minima/maxima, units, and missing optional diagnostics. It also
summarizes `stats/explained_variance`, `loss/total`, `stats/entropy`, and
`stats/approx_kl` when available. Charts are saved with the noninteractive Agg
backend. Inspect generated images visually before publishing.

## Wrapper overrides

All wrappers forward extra flags and accept `PYTHON`.

- Evaluation/GIF: `CONFIG`, `CHECKPOINT`, and `OUT`.
- Charts: `RUN`, `OUT_DIR`, `TITLE`, and optional `SUMMARY`.

Examples:

```bash
PYTHON=/path/to/fresh-uv-env/bin/python bash PPO/utils/evaluate.sh
OUT=PPO/logs/heldout.json bash PPO/utils/evaluate.sh --seed 2000
bash PPO/utils/make_charts.sh --run /path/to/run-ID.wandb --smooth 5
```
