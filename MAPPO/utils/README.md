# MAPPO evaluation and README assets

Optional helpers for MultiWalker checkpoint evaluation, GIFs, and training charts.
The default MAPPO policy uses **continuous actions**, a shared local-observation
actor, and a centralized critic. Helpers never use a scripted controller, policy
hint, reward shaping, or additional environment state as an actor input.

## Fresh uv environment

Choose an unused custom environment name for each validation run:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.venvs/mappo-heldout"
uv venv "$UV_PROJECT_ENVIRONMENT"
uv sync
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

Runtime uses the project dependencies, including PettingZoo, Gymnasium's Box2D
extra, PyTorch, NumPy, pygame, W&B, Matplotlib, imageio, and Pillow. No separate
PettingZoo MPE or SISL extra is needed for these helpers. All learned-policy
inference is on CPU. Run module commands from the repository root; wrappers
change to that root and respect `PYTHON`, defaulting to the activated `python`.
Only load trusted PyTorch checkpoints.

## Unfiltered fresh-seed evaluation

```bash
uv run python -m MAPPO.utils.evaluate \
  --config MAPPO/configs/multiwalker_delivery_long_credit.yaml \
  --episodes 30 --seed 10000 --out MAPPO/logs/evaluation.json

# Seeded random-action baseline on the same environment parameters
uv run python -m MAPPO.utils.evaluate --random \
  --episodes 30 --seed 10000 --out MAPPO/logs/random_evaluation.json

# Sample learned actions instead of deterministic policy means
uv run python -m MAPPO.utils.evaluate --stochastic \
  --episodes 30 --seed 10000 --out MAPPO/logs/sampled_evaluation.json

bash MAPPO/utils/evaluate.sh --episodes 30 --seed 10000
```

Defaults are the shown config, 30 episodes, seed 10000, and
`MAPPO/logs/evaluation.json`. `--checkpoint PATH` is optional: when absent it is
read from the **selected config's** `inference_model_path`. `--random` loads no
checkpoint. It is mutually exclusive with `--stochastic`.

Factory arguments preserve `n_walkers`, action discretization/bins, render mode,
and `cfg.env_kwargs`. Those kwargs control native MultiWalker parameters such as
`max_cycles`; the factory default remains 500. `--max-steps` optionally adds a
safety cap but does not change the native environment horizon. Episodes stop
when no active agents remain, native end flags remove the final agents, or the
safety cap is reached. Nonempty final observations do not keep already-ended
agents active.

Every episode seeds the environment and each individual action space with
`episode_seed + agent_rank`. Policy/model indexing always follows the original
`possible_agents` order. Removed agents get zero-padded observations for critic
shape consistency; **only active agent IDs receive actions**. Each deployed actor
sees its own observation. Concatenated padded observations are used only for the
centralized critic and initial value diagnostics, never as extra actor inputs.
The read-only `agent.values()` path collects initial critic values without
sampling actions. No optimization or checkpoint writes occur.

The JSON retains **every requested episode**, with:

- Native team and individual returns, discounted counterparts, lengths, seeds,
  and initial per-agent critic values.
- Native team reward defined at each parallel step as the mean over the
  **original fixed N agents**, with zero for missing rewards. Removing an agent
  never changes the score denominator.
- Per-agent last termination/truncation/removal/cap flags, active-agent patterns,
  and native infos as reported. Optional package-position/displacement diagnostics
  come from the simulator only for reporting, never from added actor inputs.
  Unavailable coordinates remain unknown; no distance is inferred from observations.
- Aggregate count, mean, population std, range, individual summaries, and survival
  at the native horizon. Discounting starts at `gamma**0` for the first reward.
- Checkpoint SHA-256 and metadata, and separate effective evaluation versus
  checkpoint-saved config provenance. Redaction is recursive. Differences are
  listed; legacy checkpoints without recorded config explicitly say
  `legacy_unknown`, rather than inventing their training settings.

**Native delivery is measured separately from survival.** The read-only
`MAPPO.mappo.diagnostics.task_status(env, initial_package_x=None)` uses the
simulator's package position, failure state, and native `get_last_dones()`.
The goal is exactly `(engine.terrain_length - 10) * (14/30)`, matching PettingZoo's
`TERRAIN_GRASS` and `TERRAIN_STEP`. The default terrain length 200 gives a goal at
approximately 88.6667 native x units. Helpers never shorten terrain, move the
goal, or turn surviving a time limit into delivery.

Failure takes precedence: package drop/game-over, a fallen walker when
`terminate_on_fall` is enabled, or crossing the left edge. Delivery requires
package x **strictly greater than** the native goal and all native done values
true with no failure. Parallel termination flags alone are not delivery evidence.
When the engine or required evidence is unavailable, `delivered` remains null;
legacy engine-less stubs retain `success=None`.

Every record reports `delivered`, `terminal_reason`, `goal_x`, signed remaining
distance, progress from the observed initial package position, package samples,
net displacement, and average speed in native x units per simulated second.
Samples have increasing time indices every 20 parallel steps plus both endpoints;
backtracking is retained. Simulator time uses its native 50 frames per second.
Optional motion/stall windows use an explicit 0.01 native-units/second threshold,
which is a motion diagnostic, never a delivery classifier.

Evaluation keeps all requested episodes. `delivery_rate` uses all episodes when
all outcomes are known; otherwise it is null and `known_delivery_rate` is labeled
separately. Outcome counts, remaining distance, speed, and progress summaries are
logged alongside native returns. The legacy generic `success` field stays null.
`survived_horizon` is still a separate no-failure horizon observation, not delivery.

For a full-horizon control, set **the selected config's**
`env_kwargs.max_cycles: 5000` and pass `--max-steps 5000`. A larger helper cap than
the native config horizon raises instead of silently extending the environment.
For example, after selecting a config with that explicit native horizon:

```bash
uv run python -m MAPPO.utils.evaluate \
  --config /path/to/full-horizon-config.yaml --checkpoint /path/to/best.pt \
  --episodes 30 --seed 10000 --max-steps 5000 \
  --out MAPPO/logs/full_horizon_evaluation.json
```

One `length` unit is a simultaneous **parallel environment step**, not one agent
transition. `agent_transitions` separately sums the active-agent count over those
steps. For example, three active agents produce three transitions in one parallel
step. Fresh seeds test the same configured environment with different reset
randomness; they do not establish transfer to different physics or team sizes.
Check that the held-out seed range was not used during training.

Evaluation forces W&B `offline`, overriding inherited online or disabled mode.
All individual and team returns plus aggregate metrics are written to the local
run, with no W&B login or cloud synchronization. Initial values are null for a
random baseline. Deterministic continuous inference uses the tanh-transformed
Gaussian mean; `--stochastic` samples the learned distribution instead.

## Deterministic continuous GIF

```bash
uv run python -m MAPPO.utils.make_gif \
  --config MAPPO/configs/multiwalker_delivery_long_credit.yaml \
  --episodes 8 --keep-top 3 --fps 30 --stride 2 --seed 30000 \
  --out MAPPO/assets/mappo_multiwalker.gif

bash MAPPO/utils/make_gif.sh
```

The shown parameters are defaults. This GIF helper requires continuous action
spaces (`discretize_actions: false`). It loads the config's checkpoint, normally
`best.pt`, renders headlessly with `SDL_VIDEODRIVER=dummy`/`rgb_array`, copies
frames before stepping, includes the final state, and briefly holds the final
frame between episodes. It handles removed agents without shifting policy IDs.

All episode team returns and native outcomes are printed. If any episode truly
delivers, only delivered episodes are eligible for selection, ranked by native
team return. Otherwise the helper can generate a clearly labeled non-delivery
control. Add `--require-delivery` to fail without writing a GIF when no episode
reaches the unchanged native goal with the required native end state.

Each selected episode includes the **entire rollout from reset through its native
endpoint**, with only frame stride/fps adjustments and a final-frame hold. There
is no temporal cut that hides stalled motion or omits transport. For a delivery
showcase, using a config with `env_kwargs.max_cycles: 5000`:

```bash
uv run python -m MAPPO.utils.make_gif \
  --config /path/to/full-horizon-config.yaml --checkpoint /path/to/best.pt \
  --episodes 8 --keep-top 3 --seed 30000 --max-steps 5000 --require-delivery \
  --out MAPPO/assets/mappo_multiwalker.gif
```

Selection still makes the GIF a showcase, not an unbiased evaluation. Use
`evaluate.py` for reported delivery rates. Positive numeric arguments are
validated; keep-top cannot exceed episodes. Inspect the rendered full episode
before publishing. Generation and native delivery flags do not replace visual
verification or runtime tests.

## Training charts

```bash
uv run python -m MAPPO.utils.make_charts \
  --run wandb/offline-run-EXAMPLE/run-EXAMPLE.wandb \
  --out-dir MAPPO/assets --title "MAPPO MultiWalker" \
  --summary MAPPO/logs/chart_metrics.json

RUN=wandb/offline-run-EXAMPLE/run-EXAMPLE.wandb \
  SUMMARY=MAPPO/logs/chart_metrics.json bash MAPPO/utils/make_charts.sh
```

Choose the actual MAPPO **training** datastore explicitly. The reader handles
repeated W&B `nested_key` fields, closes the datastore, and sorts rows by
`progress/global_step`, which counts parallel environment steps in this trainer.
`progress/agent_transitions` is a separate count. PNGs use Matplotlib's Agg
backend; reading local logs needs no network access.

- `chart_01.png`: matching batch critic predictions and discounted GAE-return
  targets, with native undiscounted team episode return in a separate panel.
  These are different quantities and units. Matching internal means is **not
  independent calibration proof** or evidence that GAE boundaries are correct.
- `chart_02.png`: policy/value losses, approximate KL, and entropy. For continuous
  policies, entropy is the **unsquashed Gaussian proxy**, not a claim about the
  exact tanh-transformed action entropy.
- `chart_03.png`: native fixed-team return and episode length, plus horizon
  survival when logged. Identical individual-return curves from native shared
  rewards are called out rather than drawn as three obscured lines. Without
  horizon metrics, individual returns are the fallback panel.

Required finite metrics: `charts/avg_return`, `charts/avg_length`,
`stats/value_mean`, `stats/return_target_mean`, `loss/policy`, `loss/value`,
`stats/entropy`, and `stats/approx_kl`. Missing required or matching value/target
rows fail clearly rather than producing fake or blank charts. Optional JSON
includes raw series and statistics for total loss, explained variance, native
horizon survival, learning rate, agent transitions, and individual returns when
logged. There is no success curve. `--smooth` defaults to 9 logged points and
changes display only, not source summary statistics. Inspect PNGs before publishing.

## Wrapper overrides

All wrappers accept `PYTHON` and forward CLI flags, which override defaults.
Evaluation/GIF accept `CONFIG`, `OUT`, and optional `CHECKPOINT`; unset checkpoints
come from the selected config. Charts accept `RUN`, `OUT_DIR`, `TITLE`, and
optional `SUMMARY`.

```bash
PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python" bash MAPPO/utils/evaluate.sh
CONFIG=/path/to/config.yaml bash MAPPO/utils/evaluate.sh --seed 20000
bash MAPPO/utils/make_charts.sh --run /path/to/run-ID.wandb --smooth 5
```
