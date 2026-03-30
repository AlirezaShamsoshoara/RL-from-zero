"""A3C main loop tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/a3c/test_main.py
    >>> # Run with unittest
    >>> # python -m unittest tests.a3c.test_main
"""

import logging
import os
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from gymnasium import spaces

from A3C.a3c.config import Config
import A3C.main as a3c_main


class DummyEnv:
    """Single-environment stub for A3C tests."""

    def __init__(self, obs_dim: int = 6, act_dim: int = 3, episode_length: int = 2):
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(act_dim)
        self.episode_length = episode_length
        self._step = 0
        self.close_called = False

    def reset(self, seed=None):
        self._step = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = np.full(
            self.observation_space.shape, float(self._step), dtype=np.float32
        )
        reward = -1.0
        terminated = self._step >= self.episode_length
        truncated = False
        return obs, reward, terminated, truncated, {}

    def close(self):
        self.close_called = True


class DummyVecEnv:
    """Vectorized-environment stub mimicking SyncVectorEnv with SAME_STEP autoreset."""

    def __init__(
        self,
        obs_dim: int = 6,
        act_dim: int = 3,
        num_envs: int = 2,
        episode_length: int = 3,
    ):
        self.single_observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.single_action_space = spaces.Discrete(act_dim)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_envs, obs_dim), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([act_dim] * num_envs)
        self.num_envs = num_envs
        self.episode_length = episode_length
        self._steps = np.zeros(num_envs, dtype=np.int32)
        self.close_called = False

    def reset(self, seed=None):
        self._steps[:] = 0
        obs = np.zeros(
            (self.num_envs,) + self.single_observation_space.shape,
            dtype=np.float32,
        )
        return obs, {}

    def step(self, actions):
        self._steps += 1
        obs_dim = self.single_observation_space.shape[0]
        next_obs = np.zeros((self.num_envs, obs_dim), dtype=np.float32)
        rewards = np.full(self.num_envs, -1.0, dtype=np.float32)
        terminated = self._steps >= self.episode_length
        truncated = np.zeros(self.num_envs, dtype=bool)

        for i in range(self.num_envs):
            if terminated[i]:
                self._steps[i] = 0
                next_obs[i] = np.zeros(obs_dim, dtype=np.float32)
            else:
                next_obs[i] = np.full(obs_dim, float(self._steps[i]), dtype=np.float32)

        return next_obs, rewards, terminated, truncated, {}

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


def _null_logger() -> logging.Logger:
    logger = logging.getLogger("a3c-test")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _make_config(**overrides) -> Config:
    cfg = Config()
    cfg.total_steps = 20
    cfg.num_workers = 1
    cfg.t_max = 5
    cfg.hidden_sizes = [8]
    cfg.activation = "relu"
    cfg.device = "cpu"
    cfg.checkpoint_interval = 100
    cfg.save_best = False
    cfg.log_interval = 10
    cfg.log_to_console = False
    cfg.log_to_file = False
    cfg.wandb_key = ""
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class TestMain(unittest.TestCase):
    def test_train_runs_with_stub_env_and_wandb(self) -> None:
        cfg = _make_config()
        dummy_wandb = DummyWandb()
        dummy_env = DummyEnv(obs_dim=6, act_dim=3, episode_length=3)

        with patch("A3C.main.Config.from_yaml", return_value=cfg), \
             patch("A3C.main.wandb", dummy_wandb), \
             patch("A3C.main.setup_logger", return_value=_null_logger()), \
             patch("A3C.main.gym.make", return_value=dummy_env), \
             patch("A3C.a3c.worker.make_env", return_value=dummy_env):
            a3c_main.train(config="unused.yaml", wandb_key="")

        self.assertIsNotNone(dummy_wandb.init_kwargs)

    def test_train_uses_env_wandb_key_when_cli_key_missing(self) -> None:
        cfg = _make_config()
        dummy_wandb = DummyWandb()
        dummy_env = DummyEnv(obs_dim=6, act_dim=3, episode_length=3)

        with patch.dict(os.environ, {"WANDB_API_KEY": "env-key"}, clear=False), \
             patch("A3C.main.Config.from_yaml", return_value=cfg), \
             patch("A3C.main.wandb", dummy_wandb), \
             patch("A3C.main.setup_logger", return_value=_null_logger()), \
             patch("A3C.main.gym.make", return_value=dummy_env), \
             patch("A3C.a3c.worker.make_env", return_value=dummy_env):
            a3c_main.train(config="unused.yaml", wandb_key="")

        self.assertIn("env-key", dummy_wandb.login_keys)

    def test_train_multi_env_config_accepted(self) -> None:
        """Verify train() accepts num_envs > 1 and initialises wandb."""
        cfg = _make_config(num_envs=2, total_steps=30)
        dummy_wandb = DummyWandb()
        dummy_env = DummyEnv(obs_dim=6, act_dim=3, episode_length=3)

        with patch("A3C.main.Config.from_yaml", return_value=cfg), \
             patch("A3C.main.wandb", dummy_wandb), \
             patch("A3C.main.setup_logger", return_value=_null_logger()), \
             patch("A3C.main.gym.make", return_value=dummy_env):
            a3c_main.train(config="unused.yaml", wandb_key="")

        self.assertIsNotNone(dummy_wandb.init_kwargs)
        self.assertEqual(dummy_wandb.init_kwargs["config"]["num_envs"], 2)

    def test_worker_multi_env_collects_episodes(self) -> None:
        """Run worker_process directly (same process) with a vectorised env stub."""
        import torch.multiprocessing as mp
        from A3C.a3c.agent import A3CAgent
        from A3C.a3c.worker import worker_process

        cfg = _make_config(num_envs=2, total_steps=20, t_max=5)
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        act_space = spaces.Discrete(3)
        agent = A3CAgent(
            obs_space=obs_space, act_space=act_space,
            hidden_sizes=[8], activation="relu",
            learning_rate=1e-3, entropy_coef=0.01,
            value_loss_coef=0.5, max_grad_norm=40.0, device="cpu",
        )
        dummy_vec_env = DummyVecEnv(obs_dim=6, act_dim=3, num_envs=2, episode_length=3)
        ctx = mp.get_context("spawn")
        global_step = ctx.Value("i", 0)
        result_queue = ctx.Queue()
        stop_event = ctx.Event()

        with patch("A3C.a3c.worker.make_vec_env", return_value=dummy_vec_env):
            worker_process(0, cfg, agent, global_step, result_queue, stop_event)

        self.assertTrue(dummy_vec_env.close_called)
        # Should have collected at least one episode report
        messages = []
        while not result_queue.empty():
            messages.append(result_queue.get_nowait())
        episode_msgs = [m for m in messages if m["kind"] == "episode"]
        update_msgs = [m for m in messages if m["kind"] == "update"]
        self.assertGreater(len(episode_msgs), 0)
        self.assertGreater(len(update_msgs), 0)

    def test_demo_runs_with_stub_env(self) -> None:
        cfg = _make_config(episodes=2, render_mode=None)
        dummy_env = DummyEnv(obs_dim=6, act_dim=3, episode_length=2)

        with patch("A3C.main.Config.from_yaml", return_value=cfg), \
             patch("A3C.main.gym.make", return_value=dummy_env), \
             patch("A3C.main.load_checkpoint") as load_ckpt, \
             patch("A3C.main.setup_logger", return_value=_null_logger()):
            a3c_main.demo(config="unused.yaml", model_path="model.pt", episodes=2)

        self.assertTrue(load_ckpt.called)
        self.assertTrue(dummy_env.close_called)


if __name__ == "__main__":
    unittest.main()
