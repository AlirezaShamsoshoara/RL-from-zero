"""MAPPO main loop tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/mappo/test_main.py
    >>> # Run with unittest
    >>> # python -m unittest tests.mappo.test_main
"""

import logging
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


def _ensure_pettingzoo_stub() -> None:
    try:
        import pettingzoo.sisl.multiwalker_v9  # noqa: F401
        return
    except Exception:
        pettingzoo = types.ModuleType("pettingzoo")
        sisl = types.ModuleType("pettingzoo.sisl")
        multiwalker_v9 = types.ModuleType("pettingzoo.sisl.multiwalker_v9")

        def _parallel_env(**kwargs):
            raise RuntimeError("pettingzoo not available in tests")

        multiwalker_v9.parallel_env = _parallel_env
        sisl.multiwalker_v9 = multiwalker_v9
        pettingzoo.sisl = sisl
        sys.modules["pettingzoo"] = pettingzoo
        sys.modules["pettingzoo.sisl"] = sisl
        sys.modules["pettingzoo.sisl.multiwalker_v9"] = multiwalker_v9


_ensure_pettingzoo_stub()

from MAPPO.mappo.config import Config
import MAPPO.main as mappo_main


class DummyObsSpace:
    def __init__(self, shape):
        self.shape = shape


class DummyActSpace:
    def __init__(self, n):
        self.n = n


class DummyParallelEnv:
    def __init__(self, obs_dim: int, act_dim: int, n_agents: int = 2, episode_length: int = 2):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.episode_length = episode_length
        self._step = 0
        self.close_called = False

    def observation_space(self, agent_id):
        return DummyObsSpace((self.obs_dim,))

    def action_space(self, agent_id):
        return DummyActSpace(self.act_dim)

    def reset(self, seed=None):
        self._step = 0
        obs = {agent_id: np.zeros(self.obs_dim, dtype=np.float32) for agent_id in self.possible_agents}
        return obs, {}

    def step(self, actions):
        self._step += 1
        rewards = {agent_id: 1.0 for agent_id in self.possible_agents}
        terminations = {
            agent_id: self._step >= self.episode_length for agent_id in self.possible_agents
        }
        truncations = {agent_id: False for agent_id in self.possible_agents}
        if self._step >= self.episode_length:
            next_obs = {}
        else:
            next_obs = {
                agent_id: np.zeros(self.obs_dim, dtype=np.float32) for agent_id in self.possible_agents
            }
        return next_obs, rewards, terminations, truncations, {}

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

    def init(self, **kwargs):
        return DummyWandb._Run()

    def log(self, data):
        self.logged.append(data)

    def login(self, key=None):
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
    logger = logging.getLogger("mappo-test")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


class TestMain(unittest.TestCase):
    def test_train_runs_with_stub_env_and_wandb(self) -> None:
        cfg = Config()
        cfg.total_timesteps = 2
        cfg.rollout_steps = 2
        cfg.update_iterations = 1
        cfg.minibatch_size = 2
        cfg.n_agents = 2
        cfg.n_walkers = 2
        cfg.actor_hidden_sizes = [8]
        cfg.critic_hidden_sizes = [8]
        cfg.activation = "tanh"
        cfg.device = "cpu"
        cfg.checkpoint_interval = 100
        cfg.save_best = False
        cfg.log_to_console = False
        cfg.log_to_file = False
        cfg.wandb_key = ""

        dummy_env = DummyParallelEnv(obs_dim=3, act_dim=2, n_agents=2, episode_length=1)
        dummy_wandb = DummyWandb()

        with patch("MAPPO.main.Config.from_yaml", return_value=cfg), patch(
            "MAPPO.main.make_multiwalker_env", return_value=dummy_env
        ), patch("MAPPO.main.wandb", dummy_wandb), patch(
            "MAPPO.main.setup_logger", return_value=_null_logger()
        ), patch("MAPPO.main.tqdm", DummyTqdm):
            mappo_main.train(config="unused.yaml", wandb_key="")

        self.assertGreaterEqual(len(dummy_wandb.logged), 1)
        self.assertTrue(dummy_env.close_called)

    def test_demo_runs_with_stub_env(self) -> None:
        cfg = Config()
        cfg.episodes = 2
        cfg.n_agents = 2
        cfg.n_walkers = 2
        cfg.actor_hidden_sizes = [8]
        cfg.critic_hidden_sizes = [8]
        cfg.activation = "tanh"
        cfg.device = "cpu"
        cfg.render_mode = None
        cfg.log_to_console = False
        cfg.log_to_file = False

        dummy_env = DummyParallelEnv(obs_dim=3, act_dim=2, n_agents=2, episode_length=1)

        with patch("MAPPO.main.Config.from_yaml", return_value=cfg), patch(
            "MAPPO.main.make_multiwalker_env", return_value=dummy_env
        ), patch("MAPPO.main.load_checkpoint") as load_ckpt, patch(
            "MAPPO.main.setup_logger", return_value=_null_logger()
        ):
            mappo_main.demo(config="unused.yaml", model_path="model.pt", episodes=2)

        self.assertTrue(load_ckpt.called)
        self.assertTrue(dummy_env.close_called)


if __name__ == "__main__":
    unittest.main()
