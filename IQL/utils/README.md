# IQL utils - README asset generators

Helpers that produce the demo assets in `IQL/assets/`:

- **`make_gif.py` / `make_gif.sh`** - render a trained IQL policy to an animated
  GIF (deterministic rollouts, keep the best, stitch).
- **`make_charts.py` / `make_charts.sh`** - read a local wandb run and write the
  three training charts (`chart_01/02/03.png`).

The `.sh` wrappers cd to the repo root, apply sensible defaults, and forward any
extra flags to the Python module.

## Requirements
Project deps plus `imageio` + `pillow` (GIF) and `matplotlib` (charts). These are
in `pyproject.toml`, so `uv sync` covers them; the `rlhero` conda env has them too.

## GIF generator
```bash
# Default: mixed-dataset Pendulum checkpoint (the showcase)
PYTHON=~/.conda/envs/rlhero/bin/python ./IQL/utils/make_gif.sh

# Random-dataset checkpoint
CONFIG=IQL/configs/pendulum_random.yaml \
  CHECKPOINT=IQL/checkpoints_random/best.pt \
  OUT=IQL/assets/iql_pendulum_random.gif \
  PYTHON=~/.conda/envs/rlhero/bin/python ./IQL/utils/make_gif.sh
```

### Parameters (`make_gif.py`)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--config` | *(required)* | YAML config (env id, network sizes, IQL hyperparams). |
| `--checkpoint` | *(required)* | `.pt` to load. Use `best.pt`, not the last update. |
| `--out` | *(required)* | Output GIF path. |
| `--episodes` | `8` | Episodes to roll out before picking the best. |
| `--keep-top` | `3` | How many top episodes to include. |
| `--fps` | `30` | Frames per second. |
| `--stride` | `2` | Keep every Nth frame to shrink the file. |
| `--seed` | `4000` | Base seed; episode i uses `seed + i`. |

## Charts generator
```bash
RUN=wandb/offline-run-XXXX/run-XXXX.wandb TITLE="IQL Pendulum (mixed)" \
  BEHAVIOR=-497 OPTIMAL=-150 \
  PYTHON=~/.conda/envs/rlhero/bin/python ./IQL/utils/make_charts.sh
```

### Parameters (`make_charts.py`)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--run` | *(required)* | Path to `run-<id>.wandb`. |
| `--out-dir` | *(required)* | Where to write `chart_01/02/03.png`. |
| `--title` | `""` | Suffix on each chart title. |
| `--behavior` | none | Optional behavior-policy return reference line on chart_01. |
| `--optimal` | none | Optional near-optimal return reference line on chart_01. |

The reader expects the keys IQL logs: `progress/update`, `eval/avg_return`,
`eval/std_return`, `loss/{actor,critic,value}`, `stats/mean_advantage`,
`stats/weight_mean`.

## Notes
- **Headless rendering:** classic-control draws through `pygame`, which needs a
  video driver. `SDL_VIDEODRIVER=dummy` (set by the wrapper) renders off-screen,
  so this works over SSH / on servers with no display.
- **Deterministic demo:** the GIF uses the policy mean (`agent.act(deterministic=True)`).
