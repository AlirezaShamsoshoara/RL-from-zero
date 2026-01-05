# MAPPO Tests

This folder contains unit tests that cover the core MAPPO implementation.

- `tests/mappo/test_networks.py` covers activation selection, actor logprob consistency, and centralized vs. decentralized critic output shapes.
- `tests/mappo/test_agent.py` validates action selection, shared-policy wiring, GAE computation, and update statistics/parameter changes.
- `tests/mappo/test_utils.py` checks seeding reproducibility, space dimension extraction, and checkpoint save/load behavior for shared vs. separate policies.
- `tests/mappo/test_main.py` runs train/demo with a stubbed multi-agent env, logger, tqdm, wandb, and checkpoint loader to keep loops fast and deterministic.

Run the suite with:

```bash
python -m pytest tests/mappo
```
