"""TRPO main loop tests (train + demo, discrete and continuous, wandb key).

Examples:
    >>> # python -m pytest tests/trpo/test_main.py
    >>> # python -m unittest tests.trpo.test_main
"""

import logging
import os
import unittest
from unittest.mock import patch

import numpy as np
from gymnasium import spaces

from TRPO.trpo.config import Config
import TRPO.main as trpo_main


class DummyVecEnv:
    """Vectorized-env stub with selectable action space."""

    def __init__(self, obs_dim, action_space, num_envs=2, episode_length=3):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.episode_length = episode_length
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.single_action_space = action_space
        self._steps = np.zeros(num_envs, dtype=np.int32)
        self.close_called = False

    def reset(self, seed=None):
        self._steps[:] = 0
        return np.zeros((self.num_envs, self.obs_dim), dtype=np.float32), {}

    def step(self, actions):
        self._steps += 1
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        rewards = np.ones(self.num_envs, dtype=np.float32)
        terminated = self._steps >= self.episode_length
        truncated = np.zeros(self.num_envs, dtype=bool)
        if np.any(terminated):
            self._steps[terminated] = 0
        return obs, rewards, terminated, truncated, {}

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

    def log(self, data):
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
    logger = logging.getLogger("trpo-test")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _cfg(num_envs=2):
    cfg = Config()
    cfg.num_envs = num_envs
    cfg.rollout_steps = 16
    cfg.total_timesteps = cfg.rollout_steps * num_envs * 2
    cfg.hidden_sizes = [8]
    cfg.activation = "tanh"
    cfg.device = "cpu"
    cfg.cg_iters = 3
    cfg.vf_iters = 2
    cfg.checkpoint_interval = 999
    cfg.save_best = False
    cfg.log_to_console = False
    cfg.log_to_file = False
    cfg.wandb_key = ""
    return cfg


class TestMain(unittest.TestCase):
    def _train(self, action_space, obs_dim=5, env_key=None, cli_key=""):
        cfg = _cfg()
        dummy_env = DummyVecEnv(obs_dim, action_space, num_envs=cfg.num_envs)
        dummy_wandb = DummyWandb()
        env_patch = patch.dict(os.environ, {"WANDB_API_KEY": env_key}) if env_key else None
        if env_patch:
            env_patch.start()
        try:
            with patch("TRPO.main.Config.from_yaml", return_value=cfg), patch(
                "TRPO.main.make_vec_env", return_value=dummy_env
            ), patch("TRPO.main.wandb", dummy_wandb), patch(
                "TRPO.main.setup_logger", return_value=_null_logger()
            ), patch("TRPO.main.tqdm", DummyTqdm):
                trpo_main.train(config="unused.yaml", wandb_key=cli_key)
        finally:
            if env_patch:
                env_patch.stop()
        return dummy_env, dummy_wandb

    def test_train_continuous(self) -> None:
        act = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        env, wb = self._train(act)
        self.assertGreaterEqual(len(wb.logged), 1)
        self.assertTrue(env.close_called)

    def test_train_discrete(self) -> None:
        env, wb = self._train(spaces.Discrete(3))
        self.assertGreaterEqual(len(wb.logged), 1)
        self.assertTrue(env.close_called)

    def test_train_uses_env_wandb_key_when_cli_key_missing(self) -> None:
        _, wb = self._train(spaces.Discrete(3), env_key="env-key", cli_key="")
        self.assertIn("env-key", wb.login_keys)

    def test_train_cli_key_overrides_env_key(self) -> None:
        _, wb = self._train(spaces.Discrete(3), env_key="env-key", cli_key="cli-key")
        self.assertIn("cli-key", wb.login_keys)
        self.assertNotIn("env-key", wb.login_keys)

    def test_demo_runs_discrete(self) -> None:
        cfg = _cfg(num_envs=1)
        cfg.episodes = 2
        cfg.render_mode = None
        dummy_env = DummyVecEnv(5, spaces.Discrete(3), num_envs=1, episode_length=2)
        with patch("TRPO.main.Config.from_yaml", return_value=cfg), patch(
            "TRPO.main.make_vec_env", return_value=dummy_env
        ), patch("TRPO.main.load_checkpoint") as load_ckpt, patch(
            "TRPO.main.setup_logger", return_value=_null_logger()
        ):
            trpo_main.demo(config="unused.yaml", model_path="model.pt", episodes=2)
        self.assertTrue(load_ckpt.called)
        self.assertTrue(dummy_env.close_called)


if __name__ == "__main__":
    unittest.main()
