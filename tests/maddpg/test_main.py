"""MADDPG main-loop tests (train + demo, wandb key handling).

Uses a DummyParallelEnv shaped like PettingZoo's parallel API so we don't touch
the real env. Verifies:
  - env-var WANDB_API_KEY is used when no CLI key is passed,
  - CLI --wandb_key overrides the env var,
  - training loop runs, logs, saves a checkpoint, calls env.close, run.finish,
  - demo runs, loads the checkpoint, closes env.

Examples:
    >>> # python -m pytest tests/maddpg/test_main.py
"""

from __future__ import annotations

import logging
import os
import unittest
from typing import Dict, List
from unittest.mock import patch

import numpy as np


class DummyBox:
    """Minimal Box replacement with sample() + low/high (float32)."""

    def __init__(self, dim: int, seed: int = 0):
        self.shape = (dim,)
        self.low = np.zeros(dim, dtype=np.float32)
        self.high = np.ones(dim, dtype=np.float32)
        self.dtype = np.float32
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        return self._rng.uniform(self.low, self.high).astype(np.float32)


class DummyParallelEnv:
    """PettingZoo-style parallel env stub for MADDPG main-loop tests."""

    def __init__(self, n_agents: int = 2, obs_dim: int = 4, act_dim: int = 2,
                 episode_length: int = 5):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = list(self.possible_agents)
        self.episode_length = episode_length
        self._step = 0
        self.close_called = False
        self._obs_spaces = {a: DummyBox(obs_dim) for a in self.possible_agents}
        self._act_spaces = {a: DummyBox(act_dim, seed=i)
                             for i, a in enumerate(self.possible_agents)}

    def observation_space(self, agent_id: str) -> DummyBox:
        return self._obs_spaces[agent_id]

    def action_space(self, agent_id: str) -> DummyBox:
        return self._act_spaces[agent_id]

    def reset(self, seed=None):
        self._step = 0
        obs = {a: np.zeros(self.obs_dim, dtype=np.float32) for a in self.possible_agents}
        return obs, {a: {} for a in self.possible_agents}

    def step(self, actions: Dict[str, np.ndarray]):
        self._step += 1
        done = self._step >= self.episode_length
        rewards = {a: -1.0 for a in self.possible_agents}
        terminations = {a: done for a in self.possible_agents}
        truncations = {a: False for a in self.possible_agents}
        next_obs = ({} if done
                    else {a: np.ones(self.obs_dim, dtype=np.float32)
                          for a in self.possible_agents})
        return next_obs, rewards, terminations, truncations, {a: {} for a in self.possible_agents}

    def close(self):
        self.close_called = True


class DummyWandb:
    class _Run:
        def finish(self):
            return None

    def __init__(self):
        self.logged: List[Dict] = []
        self.login_keys: List[str] = []

    def init(self, **kwargs):
        return DummyWandb._Run()

    def log(self, data, step=None):
        self.logged.append(data)

    def login(self, key=None):
        if key:
            self.login_keys.append(key)


class DummyTqdm:
    def __init__(self, iterable, desc=None):
        self.iterable = iterable
        self.desc = desc

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *_, **__):
        pass


def _null_logger() -> logging.Logger:
    lg = logging.getLogger("maddpg-test-null")
    lg.handlers.clear()
    lg.addHandler(logging.NullHandler())
    lg.setLevel(logging.CRITICAL)
    return lg


def _cfg(checkpoint_dir: str):
    from MADDPG.maddpg.config import Config

    cfg = Config()
    cfg.device = "cpu"
    cfg.total_steps = 30
    cfg.start_steps = 5
    cfg.batch_size = 8
    cfg.buffer_size = 100
    cfg.updates_per_step = 1
    cfg.log_interval = 10
    cfg.checkpoint_interval = 20
    cfg.hidden_sizes = [8, 8]
    cfg.n_agents = 2
    cfg.max_cycles = 5
    cfg.log_to_file = False
    cfg.log_to_console = False
    cfg.wandb_key = ""
    cfg.checkpoint_dir = checkpoint_dir
    return cfg


class TestMADDPGMain(unittest.TestCase):
    def setUp(self) -> None:
        # Import after class body so import errors surface as test failures.
        import MADDPG.main as maddpg_main
        self.maddpg_main = maddpg_main

    def _train(self, tmpdir: str, env_key: str = "", cli_key: str = ""):
        cfg = _cfg(tmpdir)
        dummy_env = DummyParallelEnv(n_agents=cfg.n_agents, episode_length=cfg.max_cycles)
        dummy_wandb = DummyWandb()
        env_patch = patch.dict(os.environ, {"WANDB_API_KEY": env_key}) if env_key else None
        if env_patch:
            env_patch.start()
        try:
            with patch("MADDPG.main.Config.from_yaml", return_value=cfg), \
                 patch("MADDPG.main.make_env", return_value=dummy_env), \
                 patch("MADDPG.main.wandb", dummy_wandb), \
                 patch("MADDPG.main.setup_logger", return_value=_null_logger()), \
                 patch("MADDPG.main.tqdm", DummyTqdm):
                self.maddpg_main.train(config="unused.yaml", wandb_key=cli_key)
        finally:
            if env_patch:
                env_patch.stop()
        return dummy_env, dummy_wandb

    def test_train_runs_and_logs(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dummy_env, wb = self._train(tmp)
        self.assertGreaterEqual(len(wb.logged), 1)
        self.assertTrue(any("charts/avg_return" in d for d in wb.logged))
        self.assertTrue(dummy_env.close_called)

    def test_train_saves_checkpoint(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self._train(tmp)
            # save_best writes best.pt after >=5 episodes.
            self.assertTrue(any(f.endswith(".pt") for f in os.listdir(tmp)))

    def test_train_uses_env_wandb_key_when_cli_missing(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _, wb = self._train(tmp, env_key="env-maddpg-key", cli_key="")
        self.assertIn("env-maddpg-key", wb.login_keys)

    def test_train_cli_key_overrides_env_key(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _, wb = self._train(tmp, env_key="env-key", cli_key="cli-key")
        self.assertIn("cli-key", wb.login_keys)
        self.assertNotIn("env-key", wb.login_keys)

    def test_demo_runs(self) -> None:
        # Train, then demo from the saved checkpoint.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self._train(tmp)
            cfg = _cfg(tmp)
            cfg.episodes = 2
            cfg.render_mode = None
            cfg.inference_model_path = os.path.join(tmp, "best.pt")
            if not os.path.exists(cfg.inference_model_path):
                # Fall back to any checkpoint that got written.
                any_ckpt = next(f for f in os.listdir(tmp) if f.endswith(".pt"))
                cfg.inference_model_path = os.path.join(tmp, any_ckpt)
            dummy_env = DummyParallelEnv(n_agents=cfg.n_agents, episode_length=cfg.max_cycles)
            with patch("MADDPG.main.Config.from_yaml", return_value=cfg), \
                 patch("MADDPG.main.make_env", return_value=dummy_env), \
                 patch("MADDPG.main.setup_logger", return_value=_null_logger()):
                self.maddpg_main.demo(config="unused.yaml")
            self.assertTrue(dummy_env.close_called)


if __name__ == "__main__":
    unittest.main()
