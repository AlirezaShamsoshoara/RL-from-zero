# TRPO utils - README asset generators

Helpers that produce the demo assets in `TRPO/assets/`:

- **`make_gif.py` / `make_gif.sh`** - render a trained policy to an animated GIF
  (deterministic rollouts, keep the best, stitch). Handles both continuous (Box)
  and discrete (Discrete) action spaces.
- **`make_charts.py` / `make_charts.sh`** - read a local wandb run and write the
  three training charts (`chart_01/02/03.png`).

The `.sh` wrappers cd to the repo root, apply sensible defaults, and forward any
extra flags to the Python module.

## Requirements
Project deps plus `imageio` + `pillow` (GIF) and `matplotlib` (charts). These are
in `pyproject.toml`, so `uv sync` covers them; the `rlhero` conda env has them too.

## GIF generator

```bash
# Default: BipedalWalker checkpoint
PYTHON=~/.conda/envs/rlhero/bin/python ./TRPO/utils/make_gif.sh

# Discrete Acrobot checkpoint
CONFIG=TRPO/configs/acrobot.yaml \
  CHECKPOINT=TRPO/checkpoints_acrobot/best.pt \
  OUT=TRPO/assets/trpo_acrobot.gif \
  PYTHON=~/.conda/envs/rlhero/bin/python ./TRPO/utils/make_gif.sh
```

### Parameters (`make_gif.py`)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--config` | *(required)* | YAML config (env id, network sizes, TRPO hyperparams). |
| `--checkpoint` | *(required)* | `.pt` to load. Use `best.pt`, not the last update. |
| `--out` | *(required)* | Output GIF path (parent dirs created). |
| `--episodes` | `8` | Episodes to roll out before picking the best. |
| `--keep-top` | `3` | How many top episodes to include. |
| `--fps` | `30` | Frames per second. |
| `--stride` | `2` | Keep every Nth frame to shrink the file. |
| `--seed` | `3000` | Base seed; episode i uses `seed + i`. |

## Charts generator

```bash
# Newest wandb run by default
RUN=wandb/offline-run-XXXX/run-XXXX.wandb TITLE="TRPO BipedalWalker-v3" \
  SOLVED=300 PYTHON=~/.conda/envs/rlhero/bin/python ./TRPO/utils/make_charts.sh
```

### Parameters (`make_charts.py`)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--run` | *(required)* | Path to `run-<id>.wandb`. |
| `--out-dir` | *(required)* | Where to write `chart_01/02/03.png`. |
| `--title` | `""` | Suffix on each chart title. |
| `--smooth` | `9` | Moving-average window for smoothed lines. |
| `--solved` | none | Optional solved-return reference line on chart_01. |
| `--max-kl` | none | Optional max-KL reference line on chart_03. |

The reader expects the keys TRPO logs: `progress/global_step`,
`charts/avg_return`, `charts/avg_length`, `loss/policy`, `loss/value`,
`stats/kl`, `stats/entropy`.

## Notes
- **Headless rendering:** Box2D/classic-control draw through `pygame`, which needs
  a video driver. `SDL_VIDEODRIVER=dummy` (set by the wrapper) renders off-screen,
  so this works over SSH / on servers with no display.
- **Deterministic demo:** the GIF uses `agent.act(..., deterministic=True)`, so it
  shows the policy mean (continuous) or argmax action (discrete), no sampling noise.
