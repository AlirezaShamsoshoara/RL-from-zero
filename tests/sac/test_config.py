"""Configuration validation without environments, training, or filesystem fixtures."""

import json
import os
import unittest
from unittest.mock import mock_open, patch

from SAC.sac.config import Config, _get_device


class TestConfig(unittest.TestCase):
    @staticmethod
    def _load(**values):
        data = {"device": "cpu"}
        data.update(values)
        with patch("builtins.open", mock_open(read_data=json.dumps(data))):
            return Config.from_yaml("unused.yaml")

    def test_positive_counts_and_replay_capacity_are_strict(self) -> None:
        for name in [
            "total_steps",
            "batch_size",
            "buffer_size",
            "updates_per_step",
            "best_window",
            "episodes",
            "max_steps_per_episode",
        ]:
            for value in [0, -1, True, 1.5]:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    self._load(**{name: value})
        with self.assertRaises(ValueError):
            self._load(batch_size=8, buffer_size=7)
        self.assertEqual(self._load().best_window, 20)
        self.assertEqual(self._load().max_steps_per_episode, 200)

    def test_optional_update_warmup_and_zero_intervals_preserve_intent(self) -> None:
        cfg = self._load(
            start_steps=17, update_after=None, log_interval=0, checkpoint_interval=0
        )
        self.assertEqual(cfg.update_after, 17)
        self.assertEqual(cfg.log_interval, 0)
        self.assertEqual(cfg.checkpoint_interval, 0)
        self.assertEqual(self._load(start_steps=17, update_after=0).update_after, 0)
        self.assertEqual(
            self._load(total_steps=1, start_steps=100, update_after=100).update_after,
            100,
        )
        for name in [
            "start_steps",
            "update_after",
            "log_interval",
            "checkpoint_interval",
            "seed",
        ]:
            for value in [-1, True, 1.5]:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    self._load(**{name: value})

    def test_float_parameters_are_finite_real_values_in_their_domains(self) -> None:
        for name in [
            "gamma",
            "tau",
            "actor_lr",
            "critic_lr",
            "alpha_lr",
            "target_entropy_scale",
            "max_grad_norm",
        ]:
            for value in [float("nan"), float("inf"), -1.0, True]:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    self._load(**{name: value})
        for name in [
            "gamma",
            "tau",
            "actor_lr",
            "critic_lr",
            "alpha_lr",
            "target_entropy_scale",
        ]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self._load(**{name: 0.0})
        for name in ["gamma", "tau"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self._load(**{name: 1.01})
        self.assertEqual(self._load(max_grad_norm=0.0).max_grad_norm, 0.0)

    def test_boolean_fields_do_not_accept_truthy_strings_or_integers(self) -> None:
        for name in ["save_best", "log_to_file", "log_to_console"]:
            for value in ["false", "true", 0, 1, None]:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    self._load(**{name: value})
        self.assertFalse(self._load(save_best=False).save_best)

    def test_yaml_compatibility_allows_empty_hidden_sizes_and_ignores_unknown_keys(
        self,
    ) -> None:
        cfg = self._load(
            hidden_sizes=[],
            legacy_option="ignored",
            wandb_key="not-published",
            checkpoint_dir="checkpoints/../models",
            env_kwargs={"g": 9.81},
        )
        self.assertEqual(cfg.hidden_sizes, [])
        self.assertFalse(hasattr(cfg, "legacy_option"))
        self.assertEqual(cfg.checkpoint_dir, os.path.normpath("checkpoints/../models"))
        self.assertNotIn("wandb_key", cfg.to_dict())
        snapshot = cfg.to_dict()
        snapshot["env_kwargs"]["g"] = 0.0
        self.assertEqual(cfg.env_kwargs["g"], 9.81)
        for sizes in [[0], [True], [2.5], "8"]:
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                self._load(hidden_sizes=sizes)
        with (
            patch("builtins.open", mock_open(read_data="- not-a-mapping")),
            self.assertRaises(ValueError),
        ):
            Config.from_yaml("unused.yaml")

    def test_auto_device_selects_available_cuda_or_cpu_without_allocations(
        self,
    ) -> None:
        for available, expected in [(False, "cpu"), (True, "cuda")]:
            with (
                self.subTest(available=available),
                patch("torch.cuda.is_available", return_value=available),
            ):
                self.assertEqual(_get_device("auto"), expected)
                self.assertEqual(self._load(device="auto").device, expected)
        self.assertEqual(_get_device("cpu"), "cpu")
        self.assertEqual(_get_device("cuda"), "cuda")
        with self.assertRaises(ValueError):
            _get_device("not-a-device")


if __name__ == "__main__":
    unittest.main()
