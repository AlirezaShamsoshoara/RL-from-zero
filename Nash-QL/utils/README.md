# Nash-QL utils - README asset generators

Helpers that produce the demo assets in `Nash-QL/assets/`:

- **`make_gif.py` / `make_gif.sh`** - roll out the greedy joint policy on grid soccer
  and render each step (via `grid soccerEnv.render`) into an animated GIF.
- **`make_charts.py` / `make_charts.sh`** - read a local wandb run and write the
  three training charts (`chart_01/02/03.png`).

The `.sh` wrappers cd to the repo root; the Python scripts put the
`nash_ql` package on `sys.path` themselves (the folder is hyphenated, so
it is not importable as `Nash-QL`).

## Requirements
Project deps plus `imageio` + `pillow` (GIF) and `matplotlib` (charts and the
grid soccer renderer). These are in `pyproject.toml`; the `rlhero` conda env has them.

## GIF generator
```bash
PYTHON=~/.conda/envs/rlhero/bin/python ./Nash-QL/utils/make_gif.sh --episodes 3
```

### Parameters (`make_gif.py`)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--config` | *(required)* | YAML config (env id, env_kwargs, hyperparams). |
| `--checkpoint` | *(required)* | Checkpoint with `q_tables` (use `best.pt`). |
| `--out` | *(required)* | Output GIF path. |
| `--episodes` | `3` | Episodes to concatenate into the GIF. |
| `--fps` | `4` | Frames per second (grid soccer is slow-moving, so a low fps reads well). |
| `--seed` | `0` | Base seed; episode i uses `seed + i`. |

## Charts generator
```bash
RUN=wandb/offline-run-XXXX/run-XXXX.wandb TITLE="Nash-QL grid soccer" \
  PYTHON=~/.conda/envs/rlhero/bin/python ./Nash-QL/utils/make_charts.sh
```

### Parameters (`make_charts.py`)
| Flag | Default | Meaning |
| --- | --- | --- |
| `--run` | *(required)* | Path to `run-<id>.wandb`. |
| `--out-dir` | *(required)* | Where to write `chart_01/02/03.png`. |
| `--title` | `""` | Suffix on each chart title. |
| `--smooth` | `11` | Moving-average window for smoothed lines. |

The reader expects the keys this trainer logs: `progress/episode`,
`charts/mean_return`, `charts/epsilon`, and `charts/agent<i>_mean_return`.

## Notes
- **No display needed:** grid soccer renders with matplotlib (Agg backend), so the
  GIF builds headlessly over SSH / on servers with no display.
- **Deterministic demo:** the GIF uses the greedy joint policy (epsilon = 0).
