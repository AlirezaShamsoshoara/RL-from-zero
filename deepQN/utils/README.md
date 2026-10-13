# DQN evaluation and README assets

Optional helpers for checkpoint evaluation and reproducible assets. All rewards
are **native Gymnasium rewards**, without reward shaping or reward scaling.
Run from the repository root. Shell wrappers change to the root and use `PYTHON`,
defaulting to `python` from your activated environment.

## Fresh uv environment

Choose an unused custom environment name for each validation run:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venvs/dqn-heldout"
uv venv "$UV_PROJECT_ENVIRONMENT"
uv sync
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

Existing project dependencies supply Gymnasium classic-control, NumPy, PyTorch,
W&B, Matplotlib, imageio, Pillow, and pygame. MountainCar needs no Box2D extras.
Helpers build the DQN agent on CPU, load online/target networks and saved state
through the core checkpoint loader, and put both networks in eval mode. Load
only trusted checkpoints. No training, replay insertion, parameter updates,
exploration-schedule changes, or checkpoint writes occur during evaluation.

## Unfiltered fresh-seed evaluation

```bash
uv run python -m deepQN.utils.evaluate \
  --config deepQN/configs/mountaincar_exploration.yaml \
  --episodes 100 --seed 1000 --out deepQN/logs/evaluation.json

# Same environment and seeds, but a seeded random-action policy
uv run python -m deepQN.utils.evaluate \
  --config deepQN/configs/mountaincar_exploration.yaml --random \
  --episodes 100 --seed 1000 --out deepQN/logs/random_evaluation.json

bash deepQN/utils/evaluate.sh --episodes 100 --seed 1000
bash deepQN/utils/evaluate.sh --random --out deepQN/logs/random_evaluation.json
```

Defaults are the values shown above. `--checkpoint PATH` is optional: the helper
uses the **selected config's** `inference_model_path` when it is omitted.
`--random` loads no checkpoint and reports null initial Q values. Configuration
`env_kwargs` are passed unchanged to the core environment factory.

Checkpoint actions use `agent.act(obs, epsilon=0, deterministic=True)`, without
RNG consumption or counter increments. Initial max-action online Q comes from
`agent.q_value(obs)`, which is read-only. Each episode resets with `seed + i` and
seeds the action space independently. The environment's termination or truncation
ends an episode naturally. The additional helper safety cap is
`cfg.max_steps_per_episode`, defaulting to 1000 if absent. **That cap does not
extend the native MountainCar 200-step TimeLimit**; an explicit environment
configuration would be needed to change the native limit.

The JSON includes all native returns, discounted returns, episode lengths,
seeds, initial observations/Q, maximum positions, and terminated/truncated/capped
flags. It also includes count/mean/population std/min/max, termination/truncation
rates, configuration, native/helper horizons, checkpoint SHA-256 and saved
counters, and the local W&B run location. Every episode and aggregate metrics are
logged with W&B **forced offline**, regardless of an inherited online or disabled
mode. No W&B login or cloud synchronization is performed.

The evaluator records `evaluation_config` separately from the checkpoint's saved
`checkpoint_training_config`, after recursive credential redaction. Relevant
configuration differences are reported in `config_mismatches`; the selected
config is not relabeled as the training recipe. Legacy checkpoints without a
saved config have unknown training provenance (`null`). Use a matching model
architecture and preprocessing config for faithful policy evaluation.

- Raw return: native undiscounted episode reward sum.
- Discounted return: `sum_t gamma**t * native_reward_t`, with the first reward at
  `t=0`; no bootstrapping at the evaluation horizon.
- Initial Q: `max_a online_Q(initial_observation, a)`.
- **MountainCar-v0 success: true termination, meaning the goal was reached.**
  Native returns remain negative even on successful episodes. Negative return is
  not a failure test. Truncation or a helper cap alone does not establish success.
- Goal success and maximum-position diagnostics apply only to the exact
  `MountainCar-v0` environment ID. For other environments, success is unknown
  (`null`), not inferred from termination or reward sign.

Fresh seeds vary MountainCar starting positions, not its physics or reward
function. Report this as same-environment held-out-start evaluation, not evidence
of transfer to different dynamics or environments. For performance metrics, use
all episodes from this evaluator rather than the filtered GIF episodes.

The tuned recipe enables `normalize_observations`: the agent rescales finite Box
coordinates before all action/value/update forwards. Replay and native rewards
remain unchanged. The normalization flag and exact input bounds are checkpointed;
loading with incompatible preprocessing is rejected. A legacy checkpoint without
this flag is treated as a raw-input model, not silently normalized.

The revised recipe uses temporal random-action exploration during training only.
`exploration_hold_min/max` specify its pulse duration. The start probability is
adjusted so epsilon refers to a nominal random-step fraction; measured occupancy
is logged as `charts/exploratory_step_fraction`. Greedy evaluator/GIF actions are
not held or scripted. Episodes clear pulse state, and no reward shaping is used.

## Deterministic headless GIF

```bash
uv run python -m deepQN.utils.make_gif \
  --config deepQN/configs/mountaincar_exploration.yaml \
  --out deepQN/assets/dqn_mountaincar.gif \
  --episodes 8 --keep-top 3 --fps 30 --stride 2 --seed 3000

bash deepQN/utils/make_gif.sh
```

The shown arguments are defaults. Checkpoint inference follows the config;
choose `best.pt` rather than the final snapshot for the showcase. Rendering uses
`SDL_VIDEODRIVER=dummy` and `rgb_array`. Frames are copied before every environment
step, include the final state, and briefly hold that final frame between episodes.
The same native termination/truncation and helper cap apply.

The tool prints **all native returns** and then the top-return selected returns.
It also prints goal status where defined. This GIF is a selected demonstration,
not an unbiased success-rate estimate. All positive arguments are validated, and
`--keep-top` cannot exceed `--episodes`. Inspect the generated GIF before publishing.

## Native-reward training charts

```bash
uv run python -m deepQN.utils.make_charts \
  --run wandb/offline-run-EXAMPLE/run-EXAMPLE.wandb \
  --out-dir deepQN/assets --title "DQN MountainCar-v0" \
  --env-id MountainCar-v0 --summary deepQN/logs/chart_metrics.json

RUN=wandb/offline-run-EXAMPLE/run-EXAMPLE.wandb \
  SUMMARY=deepQN/logs/chart_metrics.json bash deepQN/utils/make_charts.sh
```

Select the actual **DQN training** datastore explicitly. The reader handles the
repeated W&B `nested_key` field, closes the datastore, and sorts training rows by
`progress/step`. It never silently selects the newest run, which may belong to
another algorithm or an evaluation. Reading logs and generating charts require
no network access. PNGs use the noninteractive Agg backend.

- `chart_01.png`: selected-action online Q and bootstrapped TD target means from
  matching replay rows, plus native undiscounted episode return in a separate
  panel. The replay pair shares discounted training units, but **bootstrapped
  targets are not independent realized returns or calibration proof**.
- `chart_02.png`: Q loss (mean squared TD error), mean absolute TD error on a
  separate axis, and epsilon. There are no actor/critic losses in this DQN tool.
- `chart_03.png`: native episode return and length, plus MountainCar goal success.
  Supply the correct `--env-id` for the source run. For IDs other than exactly
  `MountainCar-v0`, the lower panel reports termination rate, explicitly not goal
  success. `--env-id` labels/selects semantics; it does not change training data.

Required finite metrics: `charts/avg_return`, `charts/avg_length`,
`charts/epsilon`, `charts/termination_rate`, `stats/q_value`,
`stats/target_value`, `stats/td_error`, and `loss/q_loss`. MountainCar charts also
require `charts/success_rate`. Missing metrics or matched replay rows fail clearly
rather than producing blank charts. Fractions must be in `[0,1]`; MSE and absolute
TD error must be nonnegative.

The optional `--summary PATH` records unsmoothed raw metric series and statistics,
units, and interpretation caveats. It also records `charts/buffer_size`,
`stats/start_q`, and `stats/discounted_return` when available. Start-state max Q
and observed discounted training return have a **policy mismatch**: Q targets a
greedy policy, while training uses epsilon behavior and finite episode horizons.
Do not report that gap as calibration proof or compare it directly with raw,
undiscounted episode return. `--smooth` defaults to 9 logged points and affects
only display curves, not the source summary. Verify generated PNGs visually.

## Wrapper overrides

All wrappers accept `PYTHON` and forward extra CLI flags, which override defaults.
Evaluation/GIF accept `CONFIG`, `OUT`, and optional `CHECKPOINT`. When the latter
is unset, the selected config supplies the checkpoint path. Charts accept `RUN`,
`OUT_DIR`, `TITLE`, `ENV_ID`, and optional `SUMMARY`.

```bash
PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python" bash deepQN/utils/evaluate.sh
CONFIG=/path/to/config.yaml bash deepQN/utils/evaluate.sh --seed 2000
bash deepQN/utils/make_charts.sh --run /path/to/run-ID.wandb --smooth 5
```
