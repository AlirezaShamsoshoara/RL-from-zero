# SAC Tests

The suite has 80 named test methods. Loop tests use native Gymnasium spaces and stub agents, replay, environments, checkpoints, W&B, logging, and progress bars. They do not run actual training or call `gym.make` without a mock.

- `test_networks.py`: physical-action Gaussian density with the existing affine Jacobian, stable tanh saturation, finite ordered bounds, activations, iterable hidden sizes, and linear-only shapes.
- `test_agent.py`: single and batched actions, deterministic RNG isolation, SAC entropy targets, target critic freezing, actor gradients, exact soft updates, owned CPU snapshots, checkpoint validation, legacy targets, and alpha optimizer identity.
- `test_replay_buffer.py`: strict dimensions and batch counts, whole-transition validation before mutation, finite float32 casts, terminal bitmasks, ring overwrite, owned samples, and seeded alignment.
- `test_main.py`: resource cleanup on setup, loop, and cleanup failures; W&B key precedence; independent random-action and update warmup; exact budgets; termination-only replay masks; native episode metrics; best-window selection; scoped final checkpoints; and bounded headless demos.
- `test_config.py`: strict counts, finite numeric domains, boolean types, optional update warmup, disabled intervals, compatible YAML loading, and automatic device selection without allocation.
- `test_utils.py`: caller-owned reset, action-space seeding, checkpoint round trips, basename paths, atomic failure preservation, safe CPU weights-only loading, and metadata redaction.
- `test_helpers.py`: complete unfiltered stub evaluation and observer diagnostics, illustrative full-span GIF selection with a tiny encoded metadata fixture, and distinct soft-Q versus native-return units without rendering charts.

Run this directory in a fresh custom-named `uv` environment with W&B configured for offline logging:

```bash
WANDB_MODE=offline UV_PROJECT_ENVIRONMENT=/tmp/glimmer-rl-sac-tests-unique uv run python -m pytest tests/sac -q
```

Use a new environment path for each run. Unit fixtures are not hardware, native physics, or learned-policy performance evidence.
