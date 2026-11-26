# TD3 native evaluation and assets

These optional helpers evaluate trusted snapshots, render full-span selected
showcases and read one exact local W&B training datastore. They do not import
training agents, optimizers, replay buffers or another algorithm's utilities.
Authoring was static read/write only. No tests, inference, environment execution,
training, rendering or chart execution were performed by the helpers author.
Runtime and visual verification are still required before publishing results.

Use the caller's fresh custom-named uv environment with existing project
dependencies. Wrappers respect `PYTHON`, change to repository root and forward
flags. They never install packages or activate a shared environment. Invoke them
with `bash`; executable bits were not changed.

## Evaluate all episodes

```bash
PYTHON=/path/to/fresh-uv-env/bin/python \
  bash TD3/utils/evaluate.sh --episodes 100 --seed 180000

python -m TD3.utils.evaluate \
  --config TD3/configs/mountaincar_balanced.yaml \
  --episodes 100 --seed 190000 --out TD3/logs/evaluation.json
```

The balanced config is the default, matching the parent training/demo profile.
The original 64-environment `TD3/configs/mountaincar_continuous.yaml` config remains
available explicitly. Omitted `--checkpoint` is `None` until it resolves
to **the selected config's `inference_model_path`**. There is no hardcoded baseline
or fallback in a wrapper. Explicit `--checkpoint` or `CHECKPOINT` overrides it.
Select fresh seeds disjoint from training and checkpoint selection; the helper
cannot prove they are held out. Episode `i` resets with `seed + i`. Defaults run
all 100 episodes to their native goal or 999-transition TimeLimit. No helper cap,
reward shaping, scaling, normalization, reset restriction or horizon override is
applied. Every timeout and failure is retained, not selected away.

Only raw `MountainCarContinuous-v0` is accepted: native two-feature observations,
one Box action in `[-1, 1]`, power `0.0015`, speed bound `0.07`, position bounds
`[-1.2, 0.6]`, goal position `0.45` and default goal velocity `0`. Its only accepted
physics constructor kwarg here is **`goal_velocity=0`**. Nonzero goal velocity and
unsupported physics/wrapper kwargs are rejected. A positive training
`env_kwargs.max_episode_steps` is removed and disclosed, never passed into native
evaluation. The registered 999-transition limit must remain unchanged.

**Physical native goal requires native `terminated` plus engine position >=0.45
and velocity >=0, with no caller cap.** Positive episode return is not a goal
predicate. Native reward is `100` on the engine's goal minus `0.1 * action^2`.
The **90 mean-return benchmark** is a distinct population-level reward benchmark,
not a physical goal threshold. Goal rate/count/failure stay `null` when source or
engine evidence is missing; summaries never turn a known subset into all-episode
rates. The installed native `step` source and physical attributes are inspected
before goal diagnostics are trusted. Unwrapped state is observer evidence only,
never extra actor information or an inference hint.

The actor and optional online Q1/Q2 critics are constructed directly on CPU from
`actor_state_dict`, `q1_state_dict` and `q2_state_dict`. Constructors and trusted
source loading use `torch.random.fork_rng(devices=[])` to preserve the caller's
CPU RNG, compatible with PyTorch >=2.0. Actions use exactly `Actor.forward` on
native observations, without sampling, pulse logic, clipping, action scripts,
extra info or policy modification. The trained deterministic Actor is unchanged.

Checkpoint loading requires trusted regular immutable files, not symlinks, with
`weights_only=True`, CPU placement, strict tensor keys/shapes/dtypes and finite
values. Serialized `action_scale` and `action_bias` must match native bounds
before loading. SHA-256 is checked before/after load and before output for both
checkpoint and network source. Embedded `config`/`training_config` is redacted
and checked for compatible fields. Legacy snapshots without embedded config have
**unknown training provenance**; supplied YAML is not proof of their history.

Optional `--source-root /path/to/trusted/stock` compiles that exact root's
`TD3/td3/networks.py` bytes without editing it, importing its training package or
writing stock `__pycache__`. This executes trusted Python, not a sandbox. It is
intended for the parent's stock-code control checkpoint with matching config.

JSON retains every native reward, action, initial/transition observation and
engine state, position/velocity trajectory, raw return, discounted return sum,
initial Q1/Q2, per-step Q predictions and matched Q/MC points. End flags,
timeout/cap status, max position/absolute velocity, action cost, seeds, source
hashes and redacted config/checkpoint metadata remain recorded. Motion diagnostics
use post-transition `|velocity| >=0.001` for 20 consecutive steps, labelled only
as sustained motion, never goal. Missing source/state makes these diagnostics
nullable. Trajectories are output evidence, never actor input.

Q heads learned noisy replay actions and a clipped noisy smoothed target policy,
with bootstrap across TimeLimit. Deterministic native reward-to-boundary has a
different policy and finite horizon. Q/MC points are labelled **policy/horizon
shift**, not exact greedy finite-horizon calibration. No calibration RMSE,
inferred overestimation or invented reward curves are reported.

Every evaluation and GIF rollout forces `WANDB_MODE=offline` and
`wandb.init(mode="offline")`, overriding inherited online/disabled mode. Individual
episodes, including failures, trajectories and aggregate metrics are logged
locally. There is no login, API client, credential read or cloud sync. Full
individual records stay in compact JSON; console output is curated returns and
summary, not a multidimensional trajectory dump. JSON has a 256 MiB guard.

## Full-span top-K GIF

```bash
bash TD3/utils/make_gif.sh --episodes 8 --keep-top 3 \
  --seed 280000 --stride 2 --fps 30 --hold-frames 12 \
  --out TD3/assets/td3_mountaincar_continuous.gif
```

All requested episodes are evaluated and logged. Top K by native return are
retained with stable ties and no goal/failure filter. Every selected segment
covers **reset through actual goal or full native TimeLimit**, including the final
frame. Only full-span stride/fps and an additional final hold change presentation.
No stalled/failure section is cut, concealed or replaced. Image labels include
selection/rank, seed, native return, complete length, physical goal flag and
timeout, stating **selected showcase, not unbiased evaluation**.

The `.json` sidecar, or `--summary`, retains all episodes and selected indices,
source frame indices, full-span endpoint checks, submitted counts and actual
**encoded frame count and duration read through Pillow**. Identical submitted
frames may merge while preserving duration. GIF timing is quantized to
centiseconds; it is not calculated from submitted counts. Native render metadata
is recorded; stride/fps changes playback speed, not trajectory span. Rendering
uses headless `SDL_VIDEODRIVER=dummy` only when the parent runs the helper.

## Three charts from one exact local training run

```bash
RUN=/path/to/closed-training-run.wandb \
  bash TD3/utils/make_charts.sh --config TD3/configs/mountaincar_balanced.yaml

python -m TD3.utils.make_charts \
  --run wandb/offline-run-EXAMPLE/run-EXAMPLE.wandb \
  --evaluation TD3/logs/evaluation.json --out-dir TD3/assets
```

Always pass the producer's exact config and closed immutable `.wandb` path.
No latest-run selection, authentication, API fetch or environment execution is
used. Flat `key` and repeated `nested_key` fields are supported.
`progress/step` and `progress/global_step` are environment-transition aliases
that must agree when both occur. Vector/optimizer/actor counts remain separate.
Known metric values must be finite. Missing/null actor loss on skipped updates is
not zero. Readers preserve valid later records and the union of keys even if the
first row has no actor update. No independently filtered Q series are zipped.

- `chart_01.png`: matched same-record replay Q1/Q2 and **target bootstrap Q**.
  `stats/target_q_value` is the smoothed twin-min next-state Q **before reward and
  mask, not the full Bellman target y**. TD diagnostics are separate. Native
  return is a separate panel, or optional all-episode matched initial-state Q/MC
  scatter labelled policy/horizon shift, not native calibration.
- `chart_02.png`: critic and actual delayed actor loss, plus actual logged critic/
  actor counts if available. `stats/actor_updated` may be a logging-window
  fraction; it is never inferred to be an update count. Missing counts/losses
  are disclosed, not synthesized.
- `chart_03.png`: raw native return/length plus optional logged physical goal
  rate. The horizontal 90 line is explicitly a **mean benchmark, not goal**.

Required series: `charts/avg_return`, `charts/avg_length`, `loss/critic_loss`,
`stats/q1_value`, `stats/q2_value`, `stats/target_q_value`, including at least one
single record with all three Q fields. Optional `stats/td_error`,
`stats/actor_updated`, `loss/actor_loss`, `progress/optimizer_updates`,
`progress/actor_updates`, `stats/critic_updates`, `stats/actor_updates`,
`charts/native_goal_rate` and `charts/goal_rate` are never fabricated. Old logs
without the required target series are rejected. Display smoothing spans logged
points only; every unsmoothed source row and matching Q record is retained in
curated chart JSON with per-series summaries, key union, missing metrics, units,
source/output hashes and optional evaluation points.

Existing outputs are rejected unless `--overwrite` is explicit. Source files,
symlink outputs, hard-link aliases and outputs that alias one another are always
protected. Positive counts/fps/stride/smoothing, seed ranges and empty data are
validated. Wrapper variables: evaluation/GIF use `CONFIG`, `CHECKPOINT`,
`SOURCE_ROOT`, `OUT`, `PYTHON`; GIF adds `SUMMARY`; charts use `RUN`, `CONFIG`,
`OUT_DIR`, `TITLE`, `SUMMARY`, `EVALUATION`, `PYTHON`. Extra CLI flags win. No wrapper
forces a baseline checkpoint. No timestamps are authored into result schemas.
