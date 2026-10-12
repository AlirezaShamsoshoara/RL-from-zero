# MADDPG utils - README asset generators

Helpers that produce the demo assets in `MADDPG/assets/`:

- **`make_gif.py` / `make_gif.sh`** - render the trained per-agent policies to an
  animated GIF (rolls out several deterministic episodes on PettingZoo's
  simple_spread parallel env, keeps the top-K by return, stitches them).
- **`make_charts.py` / `make_charts.sh`** - read a local wandb run and write the
  three training charts (`chart_01/02/03.png`).

The `.sh` wrappers cd to the repo root, apply sensible defaults, and forward any
extra flags to the Python module.

## Requirements
- Project deps (`pettingzoo[mpe]`, `torch`, `pyyaml`, `wandb`) plus `imageio` +
  `pillow` (GIF) and `matplotlib` (charts). These are in `pyproject.toml`, so
  `uv sync` covers them; the `rlhero` conda env has them too.

# GIF generator

## Quick start
From the repository root:

```bash
# Regenerate the default demo (trained simple_spread policy)
./MADDPG/utils/make_gif.sh

# If `python` isn't the rlhero env, point PYTHON at it:
PYTHON=~/.conda/envs/rlhero/bin/python ./MADDPG/utils/make_gif.sh
```

Or call the Python module directly (note `SDL_VIDEODRIVER=dummy` for headless
machines - PettingZoo MPE draws through pygame):

```bash
SDL_VIDEODRIVER=dummy python -m MADDPG.utils.make_gif \
    --config MADDPG/configs/simple_spread.yaml \
    --checkpoint MADDPG/checkpoints/best.pt \
    --out MADDPG/assets/maddpg_simple_spread.gif \
    --episodes 10 --keep-top 3
```

## Parameters (`make_gif.py`)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--config` | *(required)* | YAML config; provides `env_id`, `n_agents`, network sizes, action bounds, etc. |
| `--checkpoint` | *(required)* | `.pt` file to load. **Use `best.pt`** - MADDPG is high-variance and the last checkpoint is usually worse. |
| `--out` | *(required)* | Output GIF path (parent dirs are created). |
| `--episodes` | `10` | How many episodes to roll out before selecting the best. |
| `--keep-top` | `3` | How many of the highest-returning episodes to include. |
| `--fps` | `20` | Frames per second of the GIF. |
| `--stride` | `1` | Keep every Nth frame to shrink the file (higher = smaller/choppier). |
| `--seed` | `2000` | Base seed; episode *i* uses `seed + i` (change it for different rollouts). |

## Wrapper overrides (`make_gif.sh`)
Set these as environment variables before the command:

| Var | Default |
| --- | --- |
| `PYTHON` | `python` |
| `CONFIG` | `MADDPG/configs/simple_spread.yaml` |
| `CHECKPOINT` | `MADDPG/checkpoints/best.pt` |
| `OUT` | `MADDPG/assets/maddpg_simple_spread.gif` |
| `SDL_VIDEODRIVER` | `dummy` (headless off-screen rendering) |

Any extra flags are forwarded to `make_gif.py`:

```bash
./MADDPG/utils/make_gif.sh --episodes 20 --keep-top 3 --stride 2
```

# Charts generator

Reads a run's `.wandb` datastore file directly (works for **offline** runs too -
no network needed) and writes `chart_01.png` (centralized-critic Q vs realized
return), `chart_02.png` (actor & critic loss), and `chart_03.png` (return &
length).

## Quick start
```bash
# Uses the newest run under wandb/ by default
./MADDPG/utils/make_charts.sh

# Or pick a specific run and title
RUN=wandb/offline-run-XXXX/run-XXXX.wandb TITLE="MADDPG on simple_spread_v3" \
    ./MADDPG/utils/make_charts.sh
```

Or call the module directly:

```bash
python -m MADDPG.utils.make_charts \
    --run wandb/offline-run-XXXX/run-XXXX.wandb \
    --out-dir MADDPG/assets \
    --title "MADDPG on simple_spread_v3"
```

## Parameters (`make_charts.py`)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--run` | *(required)* | Path to a run's `.wandb` file (`wandb/<run>/run-<id>.wandb`). |
| `--out-dir` | *(required)* | Directory to write `chart_01/02/03.png` into. |
| `--title` | `""` | Suffix appended to each chart title. |
| `--smooth` | `9` | Moving-average window (in logging points) for smoothed lines. |

## Wrapper overrides (`make_charts.sh`)
| Var | Default |
| --- | --- |
| `PYTHON` | `python` |
| `RUN` | newest `wandb/*/run-*.wandb` |
| `OUT_DIR` | `MADDPG/assets` |
| `TITLE` | `MADDPG on simple_spread_v3` |

> The reader expects the metric keys this repo logs: `progress/step`,
> `charts/avg_return`, `charts/avg_length`, `stats/q_value`,
> `loss/critic_loss`, `loss/actor_loss`.

## Notes
- **Headless rendering:** PettingZoo MPE draws through `pygame`, which needs a
  video driver. `SDL_VIDEODRIVER=dummy` renders to an off-screen surface so this
  works over SSH / on servers with no display. The wrapper sets it automatically.
- **Deterministic demo:** actions use `noise=0.0, deterministic=True`, so the GIF
  shows the pure policy with no exploration noise.
