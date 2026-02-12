"""TD3 main loop tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/td3/test_main.py
    >>> # Run with unittest
    >>> # python -m unittest tests.td3.test_main
"""

import logging
import os
import unittest
from unittest.mock import patch

import numpy as np
from gymnasium import spaces

from TD3.td3.agent import TD3Stats
from TD3.td3.config import Config
import TD3.main as td3_main


class DummyEnv:
    def __init__(self, obs_dim: int, act_dim: int, episode_length: int = 2):
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-2.0, high=2.0, shape=(act_dim,), dtype=np.float32
        )
        self.episode_length = episode_length
        self._step = 0
        self.close_called = False

    def reset(self, seed=None):
        self._step = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = np.full(self.observation_space.shape, float(self._step), dtype=np.float32)
        reward = 1.0
        terminated = self._step >= self.episode_length
        truncated = False
        info = {}
        if terminated or truncated:
            info["episode"] = {"r": float(self._step), "l": self._step}
        return obs, reward, terminated, truncated, info

    def close(self):
        self.close_called = True


class DummyWandb:
    class _Run:
        def __init__(self):
            self.finished = False

        def finish(self):
            self.finished = True

    def __init__(self):
        self.logged = []
        self.init_kwargs = None
        self.login_keys = []

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return DummyWandb._Run()

    def log(self, data, step=None):
        self.logged.append((data, step))

    def login(self, key=None):
        self.login_keys.append(key)
        return None


class DummyTqdm:
    def __init__(self, iterable, desc=None):
        self.iterable = iterable
        self.desc = desc

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        return None


def _null_logger() -> logging.Logger:
    logger = logging.getLogger("td3-test")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


class TestMain(unittest.TestCase):
    def test_stats_to_dict_drops_none_actor_loss(self) -> None:
        stats = TD3Stats(critic_loss=1.0, actor_loss=None, q1_value=0.1, q2_value=0.2)
        data = td3_main._stats_to_dict(stats)

        self.assertIn("critic_loss", data)
        self.assertIn("q1_value", data)
        self.assertIn("q2_value", data)
        self.assertNotIn("actor_loss", data)

    def test_train_runs_with_stub_env_and_wandb(self) -> None:
        cfg = Config()
        cfg.total_steps = 3
        cfg.start_steps = 0
        cfg.batch_size = 2
        cfg.buffer_size = 10
        cfg.updates_per_step = 1
        cfg.hidden_sizes = [8]
        cfg.activation = "relu"
        cfg.device = "cpu"
        cfg.checkpoint_interval = 100
        cfg.save_best = False
        cfg.log_interval = 1
        cfg.log_to_console = False
        cfg.log_to_file = False
        cfg.wandb_key = ""

        dummy_env = DummyEnv(obs_dim=3, act_dim=2, episode_length=2)
        dummy_wandb = DummyWandb()

        with patch("TD3.main.Config.from_yaml", return_value=cfg), patch(
            "TD3.main.make_env", return_value=dummy_env
        ), patch("TD3.main.wandb", dummy_wandb), patch(
            "TD3.main.setup_logger", return_value=_null_logger()
        ), patch("TD3.main.tqdm", DummyTqdm):
            td3_main.train(config="unused.yaml", wandb_key="")

        self.assertGreaterEqual(len(dummy_wandb.logged), 1)
        self.assertTrue(dummy_env.close_called)

    def test_train_uses_env_wandb_key_when_cli_key_missing(self) -> None:
        cfg = Config()
        cfg.total_steps = 1
        cfg.start_steps = 1
        cfg.batch_size = 1
        cfg.buffer_size = 10
        cfg.updates_per_step = 1
        cfg.hidden_sizes = [8]
        cfg.activation = "relu"
        cfg.device = "cpu"
        cfg.checkpoint_interval = 100
        cfg.save_best = False
        cfg.log_interval = 1
        cfg.log_to_console = False
        cfg.log_to_file = False
        cfg.wandb_key = ""

        dummy_env = DummyEnv(obs_dim=3, act_dim=2, episode_length=1)
        dummy_wandb = DummyWandb()

        with patch.dict(os.environ, {"WANDB_API_KEY": "env-key"}, clear=False), patch(
            "TD3.main.Config.from_yaml", return_value=cfg
        ), patch("TD3.main.make_env", return_value=dummy_env), patch(
            "TD3.main.wandb", dummy_wandb
        ), patch("TD3.main.setup_logger", return_value=_null_logger()), patch(
            "TD3.main.tqdm", DummyTqdm
        ):
            td3_main.train(config="unused.yaml", wandb_key="")

        self.assertIn("env-key", dummy_wandb.login_keys)

    def test_demo_runs_with_stub_env(self) -> None:
        cfg = Config()
        cfg.episodes = 2
        cfg.hidden_sizes = [8]
        cfg.activation = "relu"
        cfg.device = "cpu"
        cfg.render_mode = None
        cfg.log_to_console = False
        cfg.log_to_file = False

        dummy_env = DummyEnv(obs_dim=3, act_dim=2, episode_length=1)

        with patch("TD3.main.Config.from_yaml", return_value=cfg), patch(
            "TD3.main.make_env", return_value=dummy_env
        ), patch("TD3.main.load_checkpoint") as load_ckpt, patch(
            "TD3.main.setup_logger", return_value=_null_logger()
        ):
            td3_main.demo(config="unused.yaml", model_path="model.pt", episodes=2)

        self.assertTrue(load_ckpt.called)
        self.assertTrue(dummy_env.close_called)


if __name__ == "__main__":
    unittest.main()
