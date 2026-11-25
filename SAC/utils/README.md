# Optional SAC utilities

These helpers evaluate trusted SAC snapshots, render a labelled showcase, and
read a local W&B training datastore. They do not change training code. Authoring
was static only: no tests, environment, inference, training, rendering, or chart
execution was performed. Runtime and visual verification remain required before
publishing results. Use a fresh custom-named uv environment with the project's
existing dependencies. Wrappers respect `PYTHON`, change to repository root, and
forward flags without installing packages or activating a shared environment.

Helper and wrapper config defaults use `SAC/configs/pendulum_fast.yaml`, matching
the CPU training/demo profile: 50,000 steps, hidden sizes `[128, 128]`, and batch
size 128. Native physics and rewards are unchanged. Omitted `--checkpoint` follows
that config to `SAC/checkpoints_fast/best.pt`; it is never hardcoded by a wrapper.
The original `SAC/configs/pendulum.yaml` recipe remains available explicitly.
For charts, always pass the config that actually produced the selected datastore.

## All-episode native evaluation

```bash
python -m SAC.utils.evaluate \
  --config SAC/configs/pendulum_fast.yaml \
  --episodes 30 --seed 180000 --out SAC/logs/evaluation.json

PYTHON=/path/to/fresh-uv-env/bin/python \
  bash SAC/utils/evaluate.sh --episodes 100 --seed 190000
```

Omitting `--checkpoint` uses **the selected config's `inference_model_path`**.
There is no forced baseline or fallback when that snapshot is missing. Explicit
`--checkpoint` or wrapper `CHECKPOINT` overrides it. Choose seeds disjoint from
training and checkpoint selection; the helper cannot prove they are held out.
Episode `i` resets with `seed + i`. All requested episodes are retained and all
native returns are printed. Summary standard deviation uses `ddof=0`.

Inference directly constructs `GaussianPolicy` with native observation dimension
3, action dimension 1, configured hidden sizes/activation, and action bounds
`[-2, 2]`. Optional online `QNetwork` heads are loaded only if their checkpoint
keys exist. It never constructs `SACAgent`, target networks, optimizers, replay,
workers, or shared optimizer state. CPU `torch.random.fork_rng(devices=[])`
restores constructor RNG state on PyTorch >=2.0. Action selection calls
`GaussianPolicy.deterministic`, implementing `tanh(mean) * scale + bias`; it never
calls a Gaussian sampler or receives extra environment info/hidden-state inputs.
No training global step is incremented.

Use trusted, frozen regular checkpoint files, not symlinks. Loading is CPU-only,
`weights_only=True`, with strict named tensor state loading. Serialized actor
`action_scale` and `action_bias` must match the **actual native environment**
before any model state is loaded. Non-finite tensors and incompatible dimensions
are rejected. Checkpoint SHA-256 is checked before/after load and again before
output, along with the networks source. Embedded `config`/`training_config`
fields are redacted and checked against the supplied config. Legacy checkpoints
without embedded config have **unknown training-unit provenance**; a supplied
YAML is not evidence of their historical reward semantics.

`--source-root /path/to/trusted/stock` loads exactly that root's
`SAC/sac/networks.py` rather than an already-imported local SAC package. This
executes trusted self-contained Python, not a sandbox. Source bytes are hashed,
compiled without creating `__pycache__` in the stock checkout, and never edited.
It does not install the stock package or import its Agent/training machinery.

### Native physics and boundaries

Only raw stock `Pendulum-v1` is accepted: `g=10`, mass/length 1, native torque
bounds, `dt=0.05`, and registered 200-transition TimeLimit. The only supported
physics constructor kwarg here is `g`; **`max_speed`, `max_torque`, and `dt` are
not accepted constructor kwargs**, despite the older YAML comment. Non-stock
physics and unsupported kwargs are rejected. A training
`env_kwargs.max_episode_steps` override is validated, removed, and disclosed;
evaluation always keeps the registered native limit. Optional evaluation
`--max-steps` is clipped to 200 and labelled separately as a caller diagnostic
cap. Leave it unset for full native episodes. GIFs offer no caller cap.

Native Pendulum normally ends by **truncation after 200 steps**. This is normal
completion, not failure. Unexpected end flags are preserved. No reward shaping,
reward normalization, reward scaling, custom action script, or reset-state
restriction is applied. The helper returns native undiscounted reward sum and
observed discounted reward-to-boundary, with no entropy term or bootstrap.

**There is no official Pendulum solved threshold or mission-success count.**
Optional `--diagnostic-return-threshold -200` records a clearly labelled numeric
benchmark: episode count at/above the chosen threshold and whether the all-episode
mean exceeds it. It does not mean the pendulum is solved or controlled.

Observer diagnostics read `env.unwrapped.state` for angle and angular velocity,
never reconstruct hidden angle from actor observations. Missing engine state
produces `null`, not guessed trajectories. Initial plus all post-transition
states and observations are saved. Post-transition upright occupancy requires
`|wrapped theta| <= 0.1 rad` and `|velocity| <= 0.5 rad/s`; upright time uses the
observed physics `dt`. `settle_snapshot` checks the last 20 observations only.
These thresholds are explicitly disclosed observer diagnostics, not actor
inputs, native termination conditions, or proof of lasting control.

JSON includes every episode's rewards, actions, observations, theta/velocity
trajectories, discounted reward targets, optional soft-Q predictions and matched
points, per-step absolute action, validated bounds, over-bound amount/flag, and
native torque-cost diagnostic `0.001 * torque^2`. The helper does not clip policy
actions: out-of-bounds actions raise. It also records end flags, cap/completion
status, installed engine/source provenance, redacted config/checkpoint provenance,
source hashes, and offline W&B run identity. `source_url`/`source_id` are `null`
for local sources, never invented external links.

Every evaluation and GIF rollout run forces both `WANDB_MODE=offline` and
`wandb.init(mode="offline")`, overriding inherited online or disabled mode.
Every episode and the all-episode summary are logged locally, including angle,
velocity, reward, and optional Q trajectories. There is no login, API client,
API-key read, upload, or sync.

## Full-span selected GIF

```bash
python -m SAC.utils.make_gif \
  --config SAC/configs/pendulum_fast.yaml --episodes 8 --keep-top 3 \
  --seed 280000 --fps 20 --stride 1 --hold-frames 20 \
  --out SAC/assets/sac_pendulum.gif

bash SAC/utils/make_gif.sh
```

Every episode is run and logged before final top-K ranking by native return,
with stable ties preserving episode order. Full initial plus all 200 transition
frames are captured, including the actual final state. Only top-K frame buffers
are retained to bound memory. An unexpected shorter episode aborts instead of
being cut, hidden, extended, or labelled as native completion.

Default stride 1 includes all 201 source frames. Other strides subsample across
the **entire** trajectory and always retain the initial and final frames. Final
holds are added separately, never replacing a stall/failure segment. The image
labels selection, rank, seed, native return, full episode length, and native
TimeLimit status. **This is an illustrative selected showcase, not an unbiased
performance evaluation.** All episode returns are printed and retained in the
sidecar JSON, defaulting to the GIF path with `.json` suffix or `--summary`.

The sidecar includes selected indices/seeds, all episode data, all-episode summary,
source frame indices for every segment, submitted frame counts, actual encoded
frame counts, and encoded duration read back through Pillow. Consecutive identical
frames may merge while retaining duration; GIF timing is quantized to
centiseconds. Counts/duration are not inferred from submitted frames. Missing or
nonpositive encoded durations raise. Default playback 20 fps follows physics
`dt=0.05`; native rendering metadata may report a different rate. Changing stride
or fps changes playback speed, not the trajectory span. `fps` is limited to 100.

## Local training charts and source JSON

```bash
python -m SAC.utils.make_charts \
  --run wandb/offline-run-EXAMPLE/run-EXAMPLE.wandb \
  --config SAC/configs/pendulum_fast.yaml --out-dir SAC/assets \
  --summary SAC/logs/chart_metrics.json

RUN=/path/to/SAC-training-run.wandb bash SAC/utils/make_charts.sh
```

Select the exact producer training config and a closed immutable `.wandb` file
explicitly. No newest-run discovery, online API, authentication, or environment
execution occurs. Flat and repeated `nested_key` protobuf keys are supported.
`progress/step` and the newer `progress/global_step` alias are accepted; if both
occur they must agree. Known metric values and steps must be finite numeric data.
Missing required series raises before plotting, never inventing replacement
curves. Smoothing defaults to nine logged points and affects display only.

- `chart_01.png`: same-batch/logging-window soft-Q1, soft-Q2, and entropy-bearing
  TD target means. Native episodic return is shown in a **separate panel**.
  Optional `--evaluation SAC/logs/evaluation.json` adds matched observation/action
  soft-Q versus observed native discounted reward-to-boundary points from all
  available heads/episodes, explicitly labelled as different objectives/horizons.
- `chart_02.png`: actor, twin-critic, and temperature losses; alpha; continuous
  mean log density; optional differential entropy estimate. Missing entropy is
  disclosed, not synthesized or replaced with A3C categorical entropy.
- `chart_03.png`: native undiscounted stochastic-training return and episode length.
  The 200-step length reference is a native TimeLimit, **not a solved threshold**.

Required series are `charts/avg_return`, `charts/avg_length`,
`loss/critic_loss`, `loss/actor_loss`, `loss/alpha_loss`, `stats/alpha_value`,
`stats/log_prob`, `stats/q1_mean`, `stats/q2_mean`, and `stats/target_q_mean`.
`stats/entropy` and `stats/td_error` are optional. Q1/Q2/target must occur together
on at least one matched history row. The original five-field SACStats histories
cannot supply Q curves and are rejected rather than backfilled. Source JSON
always retains all known unsmoothed numeric rows, matched Q rows, display-smoothed
return/length points, per-series summaries/units, missing optional metrics,
config provenance, source/output hashes, and optional evaluation pairs.

### Why soft Q is not native return calibration

The stock SAC critic target is

$$
y = r + \gamma(1-d)\left[\min_i Q'_i(s',a') - \alpha\log\pi(a'\mid s')\right].
$$

Here `d` is true termination, not native TimeLimit truncation. Thus soft Q models
a continuing **stochastic** policy with future entropy, while these inference
helpers use a deterministic tanh-mean policy and sum native rewards only through
the observed finite episode boundary. The objectives, policy, and horizon differ.
A soft-Q versus raw/deterministic return gap is **not proof of model
overestimation**. No calibration RMSE, success rate, or equality claim is reported.
Continuous differential entropy, estimated as `-mean(log_prob)`, may be negative;
log density includes the tanh Jacobian and action-scale correction.

Wrapper overrides: evaluation/GIF use `CONFIG`, `CHECKPOINT`, `SOURCE_ROOT`,
`OUT`, and `PYTHON`; GIF also accepts `SUMMARY`. Charts use `RUN`, `CONFIG`,
`OUT_DIR`, `TITLE`, `SUMMARY`, optional `EVALUATION`, and `PYTHON`. Extra CLI flags
win over wrapper defaults. Use `bash` to invoke wrappers; no executable-bit
change was performed. Helpers depend on existing PyTorch, Gymnasium classic
control rendering support, NumPy, YAML, W&B, imageio, Pillow, and Matplotlib.
Parent algorithm math, main README, tests, configurations, and existing assets
are intentionally outside these new utility files.
