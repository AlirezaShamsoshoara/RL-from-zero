# Optional A3C utilities

These helpers evaluate trusted A3C snapshots, render a labelled showcase, and
read local W&B training histories. They do not change training code. Use a fresh,
custom-named uv environment with the existing project dependencies installed.
The wrappers find the repository root, respect `PYTHON`, and forward all flags.
They do not activate a shared environment, install packages, or modify source snapshots.

Training and chart provenance use `A3C/configs/lunarlander.yaml`. Development
selection requires that training config explicitly. Evaluation and GIF helpers,
including their wrappers, default to `A3C/configs/lunarlander_greedy.yaml`, whose
`inference_model_path` points to `A3C/checkpoints/best_greedy.pt`. The greedy YAML
is a deployment alias with the baseline architecture, gamma, reward scaling, and
shaping unchanged, not a new hyperparameter profile. The workflow is baseline
training, explicit baseline development selection, then greedy inference. Create
the selected alias before running default inference helpers; they do not silently
fall back to `best.pt` if it is missing.

## Native evaluation

```bash
python -m A3C.utils.evaluate \
  --config A3C/configs/lunarlander_greedy.yaml \
  --episodes 30 --seed 180000 --out A3C/logs/evaluation.json

PYTHON=/path/to/fresh-uv-env/bin/python \
  bash A3C/utils/evaluate.sh --episodes 100 --seed 190000
```

Choose held-out seeds not used in training or checkpoint selection. Omitted
`--checkpoint` follows the selected YAML's `inference_model_path`, so a tuned
config does not silently reload the baseline snapshot. Explicit `--checkpoint`
or wrapper `CHECKPOINT` overrides that path. The resolved snapshot is checked
against output paths before any evaluation or GIF writing.

Any positive episode count is accepted, including 20, 30, and 100. Every requested episode is
included. Actions are categorical argmax on CPU with native observations only.
There is no stochastic policy, random baseline, reward hint, or extra-info input.
A local ActorCritic is used, not A3CAgent, SharedAdam, or a worker process. On
PyTorch versions supporting checkpoint assignment, meta-device construction
avoids RNG initialization. The compatibility path restores the caller's CPU RNG
state after isolated initialization; action selection never samples.

Only stock discrete `LunarLander-v3` is supported. The raw Gym factory adds no
custom shaping or scaling. Non-default physics kwargs are rejected. A training
`env_kwargs.max_episode_steps` override is removed and recorded explicitly;
evaluation retains Gym's registered native horizon. Optional `--max-steps` is a
separate diagnostic cap, not an environment termination, and is recorded on each
episode. Leave it unset for complete native evaluation.

JSON includes all native rewards, actions, returns, lengths, seeds, discounting
gamma, value predictions and matched discounted return-to-boundary samples,
termination/truncation/cap/failure flags, final observations, redacted training
configuration, snapshot SHA-256, and installed native step-source provenance.
Population standard deviation uses `ddof=0`. Checkpoints with embedded config
are checked against the supplied YAML. Old stock checkpoints without embedded
config are labelled as relying on caller-supplied value-unit provenance.

The `benchmark_ge_200_fraction` is a native return benchmark, **not a physical
landing rate**. Native landing is classified only when the installed Gym source
matches the inspected predicate and the unwrapped engine exposes `game_over`
and `lander.awake`: true termination without truncation/cap, no crash, asleep,
and inside the viewport. Pad location is a separate `landed_on_pad` diagnostic.
Unavailable engine/source evidence yields `null`, never a return-based guess.
Timeouts and caller caps remain explicit even when the engine is unknown.

For unshaped training, the helper converts values with `V / reward_scale`.
For shaped training, values remain labelled in training units, and native value
pairs are unavailable. Discounted targets contain observed native rewards only,
with no bootstrap. Truncated/capped targets are censored and excluded from the
matched full-return diagnostic. Even eligible samples are a policy-shift
diagnostic: stochastic training values need not equal argmax-policy returns.

Evaluation forces W&B `mode="offline"`, including when inherited mode is online
or disabled. Every episode and the complete summary are logged locally. The
helper never calls login, reads an API key, creates an online API client, or syncs.
Use only trusted snapshots. `--source-root /path/to/trusted/stock` can load the
stock `A3C/a3c/networks.py` architecture directly, with its root first on the
Python path, without installing or changing that checkout. This executes trusted
source, so it is not a sandbox for foreign code. Checkpoint loading is CPU-only
with `weights_only=True` and requires `model_state_dict`.

## Development-set checkpoint selection

Training-window `best.pt` follows stochastic training returns, which need not
rank snapshots the same way as native argmax inference. Select an inference
alias using an explicit development set, then freeze it before independent tests.

```bash
python -m A3C.utils.select_checkpoint \
  --config A3C/configs/lunarlander.yaml \
  --episodes 20 --seed 90000 \
  --summary A3C/logs/checkpoint-selection.json

PYTHON=/path/to/fresh-uv-env/bin/python \
  bash A3C/utils/select_checkpoint.sh --config A3C/configs/lunarlander.yaml
```

`--config` is required. `--checkpoint-dir` defaults to that profile's
`checkpoint_dir`; `--out` defaults to `checkpoint-dir/best_greedy.pt`. The
selector enumerates only immediate `checkpoint_*.pt` files plus `best.pt` in
that one directory. It does not recursively discover runs or mix profiles.
`--max-candidates` defaults to 100 and fails on excess candidates rather than
silently dropping snapshots. Empty directories, nonregular candidate files,
and symlinks are rejected. Use trusted, frozen snapshots; do not select while
training is changing candidate files.

Every candidate runs all requested episodes with the same development seeds,
CPU argmax actions, raw native rewards, and Gym's registered 1000-step horizon.
There is no training-time budget, caller cap, shaping, rendering, or filtered
subset. The ranking metric is the mean native undiscounted return over **all**
requested episodes, including crashes, native timeouts, and negative returns.
Native landing and return >=200 remain separate diagnostics, not selection
criteria. Ties choose the first candidate in deterministic filename order.

A single W&B run is forced offline and logs every candidate's episode statistics,
summary, and chosen config/checkpoint provenance. The JSON retains every candidate's
full raw episode records, failure flags, summary, SHA-256, embedded configuration,
and original snapshot metadata, including false completion flags. The chosen
source path, hash, and development seed range are explicit. These are selection
scores, **not independent held-out performance**. Evaluate the frozen alias on a
disjoint seed range, for example `--seed 380000`, without using those results to
rerank this development selection.

All candidate scores are written before copying the chosen file. The output is
a byte-identical clone made through a temporary file, flushed, hash-verified,
and atomically replaced. No model or optimizer state is edited or reserialized.
The alias preserves the underlying checkpoint's training config and snapshot
progress; selection is not a training completion or exact-resume claim.
Source hash changes abort copying, and temporary files are cleaned up. Outputs
cannot overwrite a candidate, the selected config, or the selection JSON;
`--out best.pt` is refused when `best.pt` is an input. Existing non-input inference
aliases can be replaced intentionally. An evaluation error aborts selection
rather than ranking a successful subset, while retaining completed score data.

The selector does not edit an inference YAML. The separate
`A3C/configs/lunarlander_greedy.yaml` keeps baseline architecture and reward
provenance and points `inference_model_path` to `best_greedy.pt`. It is the default
for demo, evaluation, and GIF inference, not the training or chart default. Use
`--checkpoint` explicitly to inspect another trusted snapshot. Wrapper overrides
are `CONFIG`, `CHECKPOINT_DIR`, `OUT`, `SUMMARY`, and `PYTHON`; extra CLI flags win.
No selection run was executed as part of authoring this helper.

## Complete-rollout GIF

```bash
python -m A3C.utils.make_gif \
  --config A3C/configs/lunarlander_greedy.yaml \
  --episodes 8 --keep-top 3 --seed 280000 \
  --fps 25 --stride 2 --hold-frames 15 \
  --out A3C/assets/a3c_lunarlander.gif

bash A3C/utils/make_gif.sh
```

All episodes run to native termination or native TimeLimit before ranking by raw
return. No custom cap is offered. Initial and post-transition frames include the
true final state. Striding retains that final frame, held for `hold-frames / fps`
seconds. Rendering is headless. The GIF itself labels top-K selection, seed,
return, full-episode length, and terminal status. **It is a selected showcase,
not an unbiased evaluation.** The sidecar JSON contains every episode and the
all-episode summary; it defaults to the output path with `.json` suffix or can be
set with `--summary`. The sidecar distinguishes `submitted_frames` from actual
`frames`/`encoded_frames`, and `duration_ms` is read from the encoded GIF. Encoders
may merge identical consecutive frames while preserving their accumulated hold
duration. Missing or invalid encoded metadata raises rather than guessing counts.
GIF rollouts also log every episode to offline W&B.
`fps`, `stride`, `hold-frames`, and `keep-top` must be positive; `keep-top` cannot
exceed `episodes`. Native rendering is 50 Hz, so 25 fps with stride 2 preserves
approximately real-time playback, apart from final holds.

## Local training charts

```bash
python -m A3C.utils.make_charts \
  --run wandb/offline-run-EXAMPLE/run-EXAMPLE.wandb \
  --config A3C/configs/lunarlander.yaml \
  --out-dir A3C/assets --title "A3C LunarLander-v3" \
  --summary A3C/logs/chart_metrics.json

RUN=/path/to/A3C-training-run.wandb bash A3C/utils/make_charts.sh
```

Use the exact training YAML and explicitly choose the A3C training datastore.
The reader handles repeated `nested_key` and flat keys, sorts by
`progress/global_step`, and needs no network or authentication. It rejects
missing required metrics and non-finite data instead of writing empty charts.
`--smooth` defaults to nine logged points and affects display only.

- `chart_01.png`: matched rollout `stats/value_mean` and `stats/return_mean` in
  scaled/shaped discounted training units, plus their batch-mean residual.
  These are aggregate diagnostics, not independently sampled calibration.
  Optional `--evaluation A3C/logs/evaluation.json` replaces the residual panel
  with matched initial-state native value/discounted-return samples from
  uncensored, unshaped episodes, labelled as an eligible subset and policy shift.
- `chart_02.png`: categorical policy loss, value-head MSE, weighted total A3C
  loss, and categorical entropy in separate panels.
- `chart_03.png`: native undiscounted training episode return, optional scaled
  `charts/avg_learning_return` in a separate panel, and episode length.

Required keys: `charts/avg_return`, `charts/avg_length`, `stats/value_mean`,
`stats/return_mean`, `loss/policy_loss`, `loss/value_loss`, `loss/total_loss`,
and `stats/entropy`. Values and targets must occur on matching rollout rows.
`stats/advantage_mean` and `charts/avg_learning_return` are optional. Source JSON
is always exported, defaulting to `out-dir/chart_metrics.json`, with unsmoothed
numeric rows, hashes, counts, units, missing optional metrics, and interpretation.
Titles wrap and legends use readable separate panels. Images use the Agg backend.
Inspect the generated images and run the parent's tests before publishing.

Wrapper overrides: evaluation/GIF use `CONFIG`, `CHECKPOINT`, `OUT`, and `PYTHON`.
Charts use `RUN`, `CONFIG`, `OUT_DIR`, `TITLE`, optional `SUMMARY`, and `PYTHON`.
CLI flags take precedence over wrapper defaults. Dependencies are already in the
project: Gymnasium with Box2D, PyTorch, NumPy, YAML, W&B, imageio, Pillow, and
Matplotlib. No test, training, environment, or rendering execution is implied by
adding these helpers.
