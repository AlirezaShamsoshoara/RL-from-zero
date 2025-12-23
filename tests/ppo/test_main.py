"""PPO main loop tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/ppo/test_main.py
    >>> # Run with unittest
    >>> # python -m unittest tests.ppo.test_main
"""

import logging
import unittest
from unittest.mock import patch

import numpy as np
import gymnasium as gym

from PPO.ppo.config import Config
import PPO.main as ppo_main


class DummyVecEnv:
    def __init__(self, obs_dim: int, act_dim: int, num_envs: int = 1, episode_length: int = 2):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.episode_length = episode_length
        self.single_observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Discrete(act_dim)
        self._step_counts = np.zeros(num_envs, dtype=np.int32)

    def reset(self, seed=None):
        self._step_counts[:] = 0
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        return obs, {}

    def step(self, actions):
        self._step_counts += 1
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        rewards = np.ones(self.num_envs, dtype=np.float32)
        terminated = self._step_counts >= self.episode_length
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos = {}
        if np.any(terminated):
            self._step_counts[terminated] = 0
        return obs, rewards, terminated, truncated, infos


class DummyWandb:
    class _Run:
        def __init__(self):
            self.finished = False

        def finish(self):
            self.finished = True

    def __init__(self):
        self.logged = []

    def init(self, **kwargs):
        return DummyWandb._Run()

    def log(self, data):
        self.logged.append(data)

    def login(self, key=None):
        return None


def _null_logger() -> logging.Logger:
    logger = logging.getLogger("ppo-test")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


class TestMain(unittest.TestCase):
    def test_train_runs_with_stub_env_and_wandb(self) -> None:
        cfg = Config()
        cfg.total_timesteps = 4
        cfg.rollout_steps = 4
        cfg.update_iterations = 1
        cfg.minibatch_size = 2
        cfg.num_envs = 1
        cfg.hidden_sizes = [8]
        cfg.activation = "tanh"
        cfg.device = "cpu"
        cfg.checkpoint_interval = 100
        cfg.save_best = False
        cfg.log_to_console = False
        cfg.log_to_file = False
        cfg.wandb_key = ""

        dummy_env = DummyVecEnv(obs_dim=4, act_dim=2, num_envs=cfg.num_envs)
        dummy_wandb = DummyWandb()

        with patch("PPO.main.Config.from_yaml", return_value=cfg), patch(
            "PPO.main.make_vec_env", return_value=dummy_env
        ), patch("PPO.main.wandb", dummy_wandb), patch(
            "PPO.main.setup_logger", return_value=_null_logger()
        ):
            ppo_main.train(config="unused.yaml", wandb_key="")

        self.assertGreaterEqual(len(dummy_wandb.logged), 1)


if __name__ == "__main__":
    unittest.main()
