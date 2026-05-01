# DDPG utils — README asset generators

Helpers that produce the demo assets in `DDPG/assets/`:

- **`make_gif.py` / `make_gif.sh`** — render a trained policy to an animated GIF
  (rolls out several deterministic episodes, keeps the best, stitches them).
- **`make_charts.py` / `make_charts.sh`** — read a local wandb run and write the
  three training charts (`chart_01/02/03.png`).

The `.sh` wrappers cd to the repo root, apply sensible defaults, and forward any
extra flags to the Python module.

## Requirements
- Project deps (`gymnasium[box2d]`, `torch`, `pyyaml`, `wandb`) plus `imageio` +
  `pillow` (GIF) and `matplotlib` (charts). These are now in `pyproject.toml`, so
  `uv sync` covers them; the `rlhero` conda env has them too. To add manually:
  ```bash
  uv pip install imageio pillow matplotlib "gymnasium[box2d]"
  ```

# GIF generator

## Quick start
From the repository root:

```bash
# Regenerate the default demo (tuned LunarLanderContinuous policy)
./DDPG/utils/make_gif.sh

# If `python` isn't the rlhero env, point PYTHON at it:
PYTHON=~/.conda/envs/rlhero/bin/python ./DDPG/utils/make_gif.sh
```

Or call the Python module directly (note `SDL_VIDEODRIVER=dummy` for headless
machines):

```bash
SDL_VIDEODRIVER=dummy python -m DDPG.utils.make_gif \
    --config DDPG/configs/lunarlander_continuous_tuned.yaml \
    --checkpoint DDPG/checkpoints_tuned/best.pt \
    --out DDPG/assets/ddpg_lunarlander.gif \
    --episodes 8 --keep-top 3
```

## Parameters (`make_gif.py`)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--config` | *(required)* | YAML config; provides `env_id`, network sizes, action bounds, etc. |
| `--checkpoint` | *(required)* | `.pt` file to load. **Use `best.pt`, not the last checkpoint** — DDPG is high-variance and the final model is usually worse. |
| `--out` | *(required)* | Output GIF path (parent dirs are created). |
| `--episodes` | `8` | How many episodes to roll out before selecting the best. |
| `--keep-top` | `3` | How many of the highest-returning episodes to include. |
| `--fps` | `30` | Frames per second of the GIF. |
| `--stride` | `2` | Keep every Nth frame to shrink the file (higher = smaller/choppier). |
| `--seed` | `2000` | Base seed; episode *i* uses `seed + i` (change it for different rollouts). |

## Wrapper overrides (`make_gif.sh`)
Set these as environment variables before the command:

| Var | Default |
| --- | --- |
| `PYTHON` | `python` |
| `CONFIG` | `DDPG/configs/lunarlander_continuous_tuned.yaml` |
| `CHECKPOINT` | `DDPG/checkpoints_tuned/best.pt` |
| `OUT` | `DDPG/assets/ddpg_lunarlander.gif` |
| `SDL_VIDEODRIVER` | `dummy` (headless off-screen rendering) |

Any extra flags are forwarded to `make_gif.py`, e.g.:

```bash
./DDPG/utils/make_gif.sh --episodes 12 --keep-top 2 --stride 1
```

# Charts generator

Reads a run's `.wandb` datastore file directly (works for **offline** runs too —
no network needed) and writes `chart_01.png` (critic Q vs. realized return),
`chart_02.png` (actor & critic loss), and `chart_03.png` (return & length).

## Quick start
```bash
# Uses the newest run under wandb/ by default
./DDPG/utils/make_charts.sh

# Or pick a specific run and title
RUN=wandb/offline-run-XXXX/run-XXXX.wandb TITLE="DDPG + target smoothing" \
    ./DDPG/utils/make_charts.sh
```

Or call the module directly:

```bash
python -m DDPG.utils.make_charts \
    --run wandb/offline-run-XXXX/run-XXXX.wandb \
    --out-dir DDPG/assets \
    --title "DDPG + target smoothing"
```

## Parameters (`make_charts.py`)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--run` | *(required)* | Path to a run's `.wandb` file (`wandb/<run>/run-<id>.wandb`). |
| `--out-dir` | *(required)* | Directory to write `chart_01/02/03.png` into. |
| `--title` | `""` | Suffix appended to each chart title (e.g. the run name). |
| `--smooth` | `9` | Moving-average window (in logging points) for smoothed lines. |

## Wrapper overrides (`make_charts.sh`)
| Var | Default |
| --- | --- |
| `PYTHON` | `python` |
| `RUN` | newest `wandb/*/run-*.wandb` |
| `OUT_DIR` | `DDPG/assets` |
| `TITLE` | `DDPG + target smoothing` |

> The reader expects the metric keys this repo logs: `progress/step`,
> `charts/avg_return`, `charts/avg_length`, `stats/q_value`,
> `loss/critic_loss`, `loss/actor_loss`.

## Notes
- **Headless rendering:** Box2D draws through `pygame`, which needs a video
  driver. `SDL_VIDEODRIVER=dummy` renders to an off-screen surface so this works
  over SSH / on servers with no display. The wrapper sets it automatically.
- **Deterministic demo:** actions use `noise=0.0, deterministic=True`, so the GIF
  shows the pure policy with no exploration noise.
