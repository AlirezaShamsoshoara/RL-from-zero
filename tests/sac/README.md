# SAC Tests

This folder contains unit tests that cover the core SAC implementation.

- `tests/sac/test_networks.py` covers activation helpers, MLP construction errors, Gaussian policy sampling and bounds, and Q-network output shapes.
- `tests/sac/test_agent.py` validates action selection, parameter updates/target soft updates, and state_dict round-trips for the SAC agent.
- `tests/sac/test_replay_buffer.py` checks replay buffer add/overwrite behavior, sampling preconditions, and returned tensor shapes/dtypes.
- `tests/sac/test_utils.py` verifies save/load checkpoint round-trips and stored metadata.
- `tests/sac/test_main.py` runs train/demo with stubbed env, logger, tqdm, and wandb to keep the loop fast and deterministic.

Run the suite with:

```bash
python -m pytest tests/sac
```
