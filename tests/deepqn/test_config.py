"""DQN configuration validation and public-config redaction."""

import os
import tempfile
import unittest

from deepQN.dqn.config import Config


class TestConfig(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "config.yaml")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(text)
            return Config.from_yaml(path)

    def test_float_strings_defaults_and_key_redaction(self):
        cfg = self.load(
            "lr: '0.001'\ngamma: '0.5'\nwandb_key: secret-test-key\ndevice: cpu\n"
        )
        self.assertEqual((cfg.lr, cfg.gamma), (0.001, 0.5))
        self.assertEqual(cfg.best_window, 5)
        self.assertEqual(cfg.max_steps_per_episode, 1000)
        self.assertEqual(cfg.device, "cpu")
        self.assertNotIn("wandb_key", cfg.to_dict())
        self.assertNotIn("secret-test-key", repr(cfg.to_dict()))

    def test_positive_counts_and_nonnegative_warmup_decay_are_validated(self):
        for field in (
            "total_steps",
            "train_freq",
            "batch_size",
            "buffer_size",
            "target_update_interval",
            "log_interval",
            "checkpoint_interval",
            "episodes",
            "best_window",
            "max_steps_per_episode",
        ):
            for value in (0, -1, 1.5):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    self.load(f"{field}: {value}\n")
        cfg = self.load("learning_starts: 0\nepsilon_decay_steps: 0\n")
        self.assertEqual((cfg.learning_starts, cfg.epsilon_decay_steps), (0, 0))
        for field in ("learning_starts", "epsilon_decay_steps"):
            with self.subTest(field=field), self.assertRaises((ValueError, TypeError)):
                self.load(f"{field}: -1\n")

    def test_invalid_ranges_rates_seeds_and_mappings_are_rejected(self):
        for text in (
            "lr: 0\n",
            "lr: .nan\n",
            "gamma: -0.1\n",
            "gamma: 1.1\n",
            "epsilon_start: 1.1\n",
            "epsilon_end: -0.1\n",
            "eval_epsilon: 1.1\n",
            "epsilon_start: 0.1\nepsilon_end: 0.2\n",
            "max_grad_norm: -1\n",
            "seed: -1\n",
            "env_kwargs: []\n",
            "- not a mapping\n",
        ):
            with self.subTest(text=text), self.assertRaises((ValueError, TypeError)):
                self.load(text)

    def test_normalization_flag_is_boolean_defaults_raw_and_is_public_config(self):
        self.assertIs(Config().normalize_observations, False)
        for value in ("true", "false"):
            cfg = self.load(f"normalize_observations: {value}\n")
            self.assertIs(cfg.normalize_observations, value == "true")
            self.assertIs(cfg.to_dict()["normalize_observations"], value == "true")
        for value in ("0", "1", "'false'", "[]"):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                self.load(f"normalize_observations: {value}\n")

    def test_mutable_model_and_environment_defaults_are_not_shared(self):
        first, second = Config(), Config()
        first.hidden_sizes.append(999)
        first.env_kwargs["custom"] = True
        self.assertNotIn(999, second.hidden_sizes)
        self.assertNotIn("custom", second.env_kwargs)


if __name__ == "__main__":
    unittest.main()
