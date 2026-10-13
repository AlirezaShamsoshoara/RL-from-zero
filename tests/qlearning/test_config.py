"""Tabular configuration validation, CPU resolution, and secret redaction."""

import os
import tempfile
import unittest

from Qlearning.ql.config import Config


class TestConfig(unittest.TestCase):
    def _load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "config.yaml")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(text)
            return Config.from_yaml(path)

    def test_yaml_parses_tabular_settings_and_cpu_device(self):
        cfg = self._load(
            "env_id: Stub-v0\nseed: 19\ntotal_episodes: 7\nbest_window: 3\n"
            "alpha: 0.5\ngamma: 0.8\ndevice: auto\n"
            "env_kwargs: {is_slippery: false, map_name: 8x8}\n"
            "checkpoint_dir: Qlearning/checkpoints_test/../checkpoints_test\n"
            "inference_model_path: Qlearning/checkpoints_test/best.pt\n"
            "wandb_key: config-test-secret\n"
        )
        self.assertEqual(cfg.env_id, "Stub-v0")
        self.assertEqual(cfg.seed, 19)
        self.assertEqual((cfg.total_episodes, cfg.best_window), (7, 3))
        self.assertEqual((cfg.alpha, cfg.gamma), (0.5, 0.8))
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.env_kwargs, {"is_slippery": False, "map_name": "8x8"})
        self.assertEqual(
            cfg.checkpoint_dir, os.path.normpath("Qlearning/checkpoints_test")
        )
        self.assertEqual(cfg.wandb_key, "config-test-secret")
        self.assertNotIn("wandb_key", cfg.to_dict())
        self.assertNotIn("config-test-secret", repr(cfg.to_dict()))
        self.assertEqual(cfg.to_dict()["best_window"], 3)

    def test_cpu_is_accepted_but_non_cpu_devices_are_rejected(self):
        self.assertEqual(self._load("device: cpu\n").device, "cpu")
        for device in ("cuda", "cuda:0", "mps", "not-a-device"):
            with (
                self.subTest(device=device),
                self.assertRaises((ValueError, TypeError)),
            ):
                self._load(f"device: {device}\n")

    def test_counts_must_be_positive_integers(self):
        for field in (
            "total_episodes",
            "max_steps_per_episode",
            "log_interval",
            "checkpoint_interval",
            "best_window",
            "episodes",
        ):
            for value in (0, -1, 1.5):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    self._load(f"{field}: {value}\n")

    def test_invalid_tabular_hyperparameters_are_rejected(self):
        texts = [
            "alpha: 0\n",
            "alpha: 1.1\n",
            "alpha: .nan\n",
            "gamma: -0.1\n",
            "gamma: 1.1\n",
            "gamma: .inf\n",
            "epsilon_start: 1.1\n",
            "epsilon_end: -0.1\n",
            "epsilon_start: 0.1\nepsilon_end: 0.2\n",
            "epsilon_decay: -0.1\n",
            "epsilon_decay: .nan\n",
        ]
        for text in texts:
            with self.subTest(text=text), self.assertRaises((ValueError, TypeError)):
                self._load(text)

    def test_gamma_endpoints_and_zero_epsilon_decay_are_valid(self):
        for gamma in (0, 1):
            with self.subTest(gamma=gamma):
                cfg = self._load(f"gamma: {gamma}\nalpha: 1\nepsilon_decay: 0\n")
                self.assertEqual(cfg.gamma, gamma)
                self.assertEqual(cfg.epsilon_decay, 0)

    def test_float_strings_are_parsed_and_invalid_seeds_or_mappings_rejected(self):
        cfg = self._load("alpha: '0.5'\ngamma: '0.8'\nepsilon_decay: '0.01'\n")
        self.assertEqual((cfg.alpha, cfg.gamma, cfg.epsilon_decay), (0.5, 0.8, 0.01))
        for text in (
            "seed: -1\n",
            "seed: 4294967296\n",
            "seed: 1.5\n",
            "env_kwargs: []\n",
            "- not\n- a mapping\n",
        ):
            with self.subTest(text=text), self.assertRaises((ValueError, TypeError)):
                self._load(text)

    def test_environment_kwargs_are_not_shared_between_configs(self):
        first, second = Config(), Config()
        first.env_kwargs["map_name"] = "8x8"
        self.assertNotIn("map_name", second.env_kwargs)


if __name__ == "__main__":
    unittest.main()
