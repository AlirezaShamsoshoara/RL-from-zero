"""IQL main loop tests (train + demo, wandb key handling).

Examples:
    >>> # python -m pytest tests/iql/test_main.py
    >>> # python -m unittest tests.iql.test_main
"""

import logging
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium import spaces

from IQL.iql.config import Config
from IQL.iql.dataset import OfflineDataset
import IQL.main as iql_main


class DummyEnv:
    def __init__(self, obs_dim=3, act_dim=1, episode_length=3):
        self.observation_space = spaces.Box(-1.0, 1.0, (obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-2.0, 2.0, (act_dim,), dtype=np.float32)
        self.episode_length = episode_length
        self._step = 0
        self.close_called = False

    def reset(self, seed=None):
        self._step = 0
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        terminated = self._step >= self.episode_length
        return obs, 1.0, terminated, False, {}

    def close(self):
        self.close_called = True


class DummyWandb:
    class _Run:
        def finish(self):
            return None

    def __init__(self):
        self.logged = []
        self.login_keys = []

    def init(self, **kwargs):
        return DummyWandb._Run()

    def log(self, data, step=None):
        self.logged.append(data)

    def login(self, key=None):
        self.login_keys.append(key)
        return None


class DummyTqdm:
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *a, **k):
        return None


def _null_logger():
    logger = logging.getLogger("iql-test")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _stub_dataset(obs_dim=3, act_dim=1, n=64):
    obs = np.random.randn(n, obs_dim).astype(np.float32)
    actions = np.tanh(np.random.randn(n, act_dim)).astype(np.float32) * 2.0
    rewards = np.random.randn(n).astype(np.float32)
    next_obs = np.random.randn(n, obs_dim).astype(np.float32)
    dones = np.zeros(n, dtype=np.float32)
    ds = OfflineDataset.from_arrays(obs, actions, rewards, next_obs, dones,
                                    device=torch.device("cpu"))
    return ds, ds.stats


def _cfg():
    cfg = Config()
    cfg.total_updates = 4
    cfg.batch_size = 16
    cfg.hidden_sizes = [8]
    cfg.activation = "relu"
    cfg.device = "cpu"
    cfg.log_interval = 1
    cfg.eval_interval = 0
    cfg.checkpoint_interval = 0
    cfg.save_best = False
    cfg.log_to_console = False
    cfg.log_to_file = False
    cfg.wandb_key = ""
    return cfg


class TestMain(unittest.TestCase):
    def _train(self, env_key=None, cli_key=""):
        cfg = _cfg()
        dummy_env = DummyEnv()
        dummy_wandb = DummyWandb()
        env_patch = patch.dict(os.environ, {"WANDB_API_KEY": env_key}) if env_key else None
        if env_patch:
            env_patch.start()
        try:
            with patch("IQL.main.Config.from_yaml", return_value=cfg), patch(
                "IQL.main.make_env", return_value=dummy_env
            ), patch("IQL.main.build_dataset", return_value=_stub_dataset()), patch(
                "IQL.main.wandb", dummy_wandb
            ), patch("IQL.main.setup_logger", return_value=_null_logger()), patch(
                "IQL.main.tqdm", DummyTqdm
            ):
                iql_main.train(config="unused.yaml", wandb_key=cli_key)
        finally:
            if env_patch:
                env_patch.stop()
        return dummy_env, dummy_wandb

    def test_train_runs(self) -> None:
        env, wb = self._train()
        self.assertGreaterEqual(len(wb.logged), 1)
        self.assertTrue(env.close_called)

    def test_train_uses_env_wandb_key_when_cli_missing(self) -> None:
        _, wb = self._train(env_key="env-key", cli_key="")
        self.assertIn("env-key", wb.login_keys)

    def test_train_cli_key_overrides_env_key(self) -> None:
        _, wb = self._train(env_key="env-key", cli_key="cli-key")
        self.assertIn("cli-key", wb.login_keys)
        self.assertNotIn("env-key", wb.login_keys)

    def test_train_eval_and_best_checkpoint(self) -> None:
        cfg = _cfg()
        cfg.total_updates = 2
        cfg.eval_interval = 1
        cfg.eval_episodes = 1
        cfg.save_best = True
        dummy_env = DummyEnv()
        dummy_wandb = DummyWandb()
        with patch("IQL.main.Config.from_yaml", return_value=cfg), patch(
            "IQL.main.make_env", return_value=dummy_env
        ), patch("IQL.main.build_dataset", return_value=_stub_dataset()), patch(
            "IQL.main.wandb", dummy_wandb
        ), patch("IQL.main.setup_logger", return_value=_null_logger()), patch(
            "IQL.main.tqdm", DummyTqdm
        ), patch("IQL.main.save_checkpoint") as save_ckpt:
            iql_main.train(config="unused.yaml", wandb_key="")
        self.assertTrue(save_ckpt.called)

    def test_demo_runs(self) -> None:
        cfg = _cfg()
        cfg.episodes = 2
        cfg.render_mode = None
        dummy_env = DummyEnv()
        with patch("IQL.main.Config.from_yaml", return_value=cfg), patch(
            "IQL.main.make_env", return_value=dummy_env
        ), patch("IQL.main.load_checkpoint") as load_ckpt, patch(
            "IQL.main.setup_logger", return_value=_null_logger()
        ):
            iql_main.demo(config="unused.yaml", model_path="model.pt", episodes=2)
        self.assertTrue(load_ckpt.called)
        self.assertTrue(dummy_env.close_called)


if __name__ == "__main__":
    unittest.main()
