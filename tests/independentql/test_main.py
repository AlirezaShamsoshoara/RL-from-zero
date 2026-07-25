"""Independent-QL main loop tests (train + demo, wandb key handling).

Examples:
    >>> # python -m pytest tests/independentql/test_main.py
"""

import logging
import os
import unittest
from unittest.mock import patch

import numpy as np

import main as iql_main
from independent_ql.config import Config
from independent_ql.envs.line_world import StepResult


class DummyEnv:
    def __init__(self, n_agents=2, n_states=12, n_actions=3, episode_length=2):
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.episode_length = episode_length
        self._steps = 0

    def reset(self, seed=None):
        self._steps = 0
        return [0 for _ in range(self.n_agents)]

    def step(self, actions):
        self._steps += 1
        truncated = self._steps >= self.episode_length
        return StepResult(
            observations=[0 for _ in range(self.n_agents)],
            rewards=[1.0 for _ in range(self.n_agents)],
            terminated=[False for _ in range(self.n_agents)],
            truncated=truncated,
            info={"positions": tuple(0 for _ in range(self.n_agents))},
        )


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
    logger = logging.getLogger("indql-test")
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _cfg():
    cfg = Config()
    cfg.total_episodes = 3
    cfg.max_steps_per_episode = 3
    cfg.log_interval = 1
    cfg.checkpoint_interval = 100
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
            with patch("main.Config.from_yaml", return_value=cfg), patch(
                "main.make_env", return_value=dummy_env
            ), patch("main.wandb", dummy_wandb), patch(
                "main.setup_logger", return_value=_null_logger()
            ), patch("main.tqdm", DummyTqdm), patch("main.save_checkpoint"):
                iql_main.train(config="unused.yaml", wandb_key=cli_key)
        finally:
            if env_patch:
                env_patch.stop()
        return dummy_wandb

    def test_train_runs_and_logs(self) -> None:
        wb = self._train()
        self.assertGreaterEqual(len(wb.logged), 1)
        # per-agent return keys are logged
        self.assertTrue(any("charts/agent0_mean_return" in d for d in wb.logged))

    def test_train_uses_env_wandb_key_when_cli_missing(self) -> None:
        wb = self._train(env_key="env-key", cli_key="")
        self.assertIn("env-key", wb.login_keys)

    def test_train_cli_key_overrides_env_key(self) -> None:
        wb = self._train(env_key="env-key", cli_key="cli-key")
        self.assertIn("cli-key", wb.login_keys)
        self.assertNotIn("env-key", wb.login_keys)

    def test_demo_runs(self) -> None:
        cfg = _cfg()
        cfg.episodes = 2
        dummy_env = DummyEnv()
        ckpt = {"q_tables": np.zeros((dummy_env.n_agents, dummy_env.n_states,
                                       dummy_env.n_actions), dtype=np.float32)}
        with patch("main.Config.from_yaml", return_value=cfg), patch(
            "main.make_env", return_value=dummy_env
        ), patch("main.load_checkpoint", return_value=ckpt) as load_ckpt, patch(
            "main.setup_logger", return_value=_null_logger()
        ):
            iql_main.demo(config="unused.yaml", model_path="model.pt", episodes=2)
        self.assertTrue(load_ckpt.called)


if __name__ == "__main__":
    unittest.main()
