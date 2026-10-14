"""Configuration contract checks use temporary YAML, not environments."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from TD3.td3.config import Config


def _load_config(values):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.yaml"
        path.write_text(yaml.safe_dump(values), encoding="utf-8")
        return Config.from_yaml(str(path))


class TestConfig(unittest.TestCase):
    def test_defaults_use_native_window_and_optional_learning_gate(self):
        cfg = _load_config({"device": "cpu"})
        self.assertEqual(cfg.best_window, 20)
        self.assertEqual(cfg.max_grad_norm, 0.0)
        self.assertIsNone(Config().update_after)
        self.assertEqual(cfg.update_after, cfg.start_steps)
        self.assertEqual(cfg.max_steps_per_episode, 999)
        self.assertEqual(cfg.render_mode, None)

    def test_positive_counts_are_strict_integers(self):
        for key in [
            "total_steps",
            "batch_size",
            "buffer_size",
            "num_envs",
            "updates_per_step",
            "policy_delay",
            "best_window",
            "episodes",
            "random_action_hold_min",
            "random_action_hold_max",
            "max_steps_per_episode",
        ]:
            for value in [0, -1, 1.5, True]:
                with (
                    self.subTest(key=key, value=value),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    _load_config({key: value, "device": "cpu"})

    def test_boolean_fields_do_not_coerce_strings_or_numbers(self):
        for key in ["save_best", "log_to_console", "log_to_file"]:
            for value in ["false", "true", 0, 1, None]:
                with (
                    self.subTest(key=key, value=value),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    _load_config({key: value, "device": "cpu"})
        cfg = _load_config(
            {
                "save_best": False,
                "log_to_console": False,
                "log_to_file": True,
                "device": "cpu",
            }
        )
        self.assertIs(cfg.save_best, False)
        self.assertIs(cfg.log_to_console, False)
        self.assertIs(cfg.log_to_file, True)

    def test_numeric_parameters_reject_nonfinite_and_out_of_bounds(self):
        for key in [
            "actor_lr",
            "critic_lr",
            "gamma",
            "tau",
            "target_noise",
            "noise_clip",
            "exploration_noise",
            "random_action_prob",
            "max_grad_norm",
        ]:
            for value in [float("nan"), float("inf"), True]:
                with (
                    self.subTest(key=key, value=value),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    _load_config({key: value, "device": "cpu"})
        for key, value in [
            ("actor_lr", 0),
            ("critic_lr", -1),
            ("gamma", 1.1),
            ("tau", 0),
            ("tau", 1.1),
            ("random_action_prob", -0.1),
            ("random_action_prob", 1.1),
            ("max_grad_norm", -1),
            ("target_noise", -1),
            ("exploration_noise", -1),
            ("noise_clip", -1),
        ]:
            with (
                self.subTest(key=key, value=value),
                self.assertRaises((ValueError, TypeError)),
            ):
                _load_config({key: value, "device": "cpu"})

    def test_empty_hidden_layers_allowed_but_bad_widths_rejected(self):
        cfg = _load_config({"hidden_sizes": [], "device": "cpu"})
        self.assertEqual(cfg.hidden_sizes, [])
        for value in [[0], [-1], [1.5], [True], "8", None]:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                _load_config({"hidden_sizes": value, "device": "cpu"})

    def test_zero_cadences_disable_without_permitting_negative_or_fractional_counts(
        self,
    ):
        cfg = _load_config(
            {
                "log_interval": 0,
                "checkpoint_interval": 0,
                "start_steps": 0,
                "update_after": 0,
                "device": "cpu",
            }
        )
        self.assertEqual(
            (
                cfg.log_interval,
                cfg.checkpoint_interval,
                cfg.start_steps,
                cfg.update_after,
            ),
            (0, 0, 0, 0),
        )
        for key in [
            "log_interval",
            "checkpoint_interval",
            "start_steps",
            "update_after",
        ]:
            for value in [-1, 1.5, True]:
                with (
                    self.subTest(key=key, value=value),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    _load_config({key: value, "device": "cpu"})

    def test_vector_budget_exact_and_warmup_is_a_requested_threshold(self):
        for values in [
            dict(total_steps=7, num_envs=2, start_steps=0),
            dict(random_action_hold_min=4, random_action_hold_max=3),
        ]:
            with (
                self.subTest(values=values),
                self.assertRaises((ValueError, TypeError)),
            ):
                _load_config({**values, "device": "cpu"})
        for start in [3, 10]:
            cfg = _load_config(
                dict(total_steps=8, num_envs=2, start_steps=start, device="cpu")
            )
            self.assertEqual(cfg.start_steps, start)
            self.assertEqual(cfg.update_after, start)
        cfg = _load_config(
            {
                "start_steps": 4,
                "update_after": 0,
                "random_action_prob": 1.0,
                "device": "cpu",
            }
        )
        self.assertEqual(cfg.start_steps, 4)
        self.assertEqual(cfg.update_after, 0)
        implicit = _load_config({"start_steps": 4, "device": "cpu"})
        self.assertEqual(implicit.update_after, implicit.start_steps)

    def test_auto_cpu_and_to_dict_are_safe_and_complete(self):
        with patch("torch.cuda.is_available", return_value=False):
            cfg = _load_config(
                {
                    "device": "auto",
                    "hidden_sizes": [],
                    "env_kwargs": {"max_episode_steps": 999},
                }
            )
        self.assertEqual(cfg.device, "cpu")
        data = cfg.to_dict()
        for key in [
            "best_window",
            "update_after",
            "max_grad_norm",
            "max_steps_per_episode",
        ]:
            self.assertIn(key, data)
        self.assertNotIn("wandb_key", data)
        data["env_kwargs"]["max_episode_steps"] = 1
        data["hidden_sizes"].append(7)
        self.assertEqual(cfg.env_kwargs["max_episode_steps"], 999)
        self.assertEqual(cfg.hidden_sizes, [])
