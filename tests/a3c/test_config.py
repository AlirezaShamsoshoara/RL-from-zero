"""Configuration validation and public serialization, with YAML reads mocked."""

from dataclasses import fields
import io
import unittest
from unittest.mock import patch

import yaml

from A3C.a3c.config import Config


def _load_config(**values):
    with patch("builtins.open", return_value=io.StringIO(yaml.safe_dump(values))):
        return Config.from_yaml("not-a-real-config.yaml")


class TestConfig(unittest.TestCase):
    def test_defaults_cpu_auto_and_independent_env_kwargs(self):
        with patch("torch.cuda.is_available", return_value=True):
            cfg = _load_config(device="auto")
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.best_window, 20)
        self.assertEqual(cfg.max_steps_per_episode, 1000)
        self.assertEqual(cfg.env_kwargs, {})
        first, second = Config(), Config()
        first.env_kwargs["custom"] = 1
        self.assertEqual(second.env_kwargs, {})

    def test_strict_positive_counts_and_vector_budget_divisibility(self):
        names = (
            "num_workers",
            "num_envs",
            "t_max",
            "total_steps",
            "episodes",
            "best_window",
            "max_steps_per_episode",
        )
        for name in names:
            for invalid in (0, -1, True, 1.5):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        _load_config(**{name: invalid})
        with self.assertRaises(ValueError):
            _load_config(total_steps=5, num_envs=2)
        self.assertEqual(_load_config(total_steps=6, num_envs=2).total_steps, 6)

    def test_strict_booleans_and_nonnegative_logging_intervals(self):
        names = (
            "save_best",
            "log_to_console",
            "log_to_file",
            "anneal_lr",
            "anneal_entropy",
            "normalize_advantages",
            "reward_shaping",
        )
        for name in names:
            for invalid in (1, 0, "false", None):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        _load_config(**{name: invalid})
        for name in ("log_interval", "checkpoint_interval"):
            self.assertEqual(getattr(_load_config(**{name: 0}), name), 0)
            for invalid in (-1, True, 1.5):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        _load_config(**{name: invalid})

    def test_numeric_fields_reject_nonfinite_and_invalid_ranges(self):
        for name in (
            "gamma",
            "learning_rate",
            "entropy_coef",
            "entropy_coef_end",
            "value_loss_coef",
            "max_grad_norm",
            "reward_scale",
            "worker_timeout",
            "shutdown_timeout",
        ):
            for invalid in (float("nan"), float("inf"), True):
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaises(ValueError):
                        _load_config(**{name: invalid})
        for values in (
            {"gamma": -0.1},
            {"gamma": 1.1},
            {"learning_rate": 0},
            {"worker_timeout": 0},
            {"shutdown_timeout": 0},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    _load_config(**values)

    def test_invalid_device_activation_hidden_sizes_and_env_kwargs(self):
        for values in (
            {"device": "cuda"},
            {"activation": "unknown"},
            {"hidden_sizes": [0]},
            {"hidden_sizes": [True]},
            {"env_kwargs": []},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    _load_config(**values)

    def test_public_config_includes_all_fields_except_credentials(self):
        cfg = _load_config(
            wandb_key="never-log-this",
            env_kwargs={"enable_wind": True},
            best_window=7,
            max_steps_per_episode=55,
        )
        public = cfg.to_dict()
        self.assertEqual(
            set(public), {field.name for field in fields(Config)} - {"wandb_key"}
        )
        self.assertNotIn("never-log-this", repr(public))
        self.assertEqual(public["env_kwargs"], {"enable_wind": True})
        self.assertEqual(public["best_window"], 7)
        self.assertEqual(public["max_steps_per_episode"], 55)
