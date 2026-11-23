"""MAPPO config validation and preserved continuous shared-policy defaults."""

import os
import tempfile
import unittest

from MAPPO.mappo.config import Config


class TestConfig(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "config.yaml")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(text)
            return Config.from_yaml(path)

    def test_defaults_are_shared_continuous_and_mutable_fields_are_independent(self):
        first, second = Config(), Config()
        self.assertTrue(first.share_policy)
        self.assertFalse(first.discretize_actions)
        self.assertEqual((first.best_window, first.max_steps_per_episode), (10, 500))
        first.env_kwargs["max_cycles"] = 13
        first.actor_hidden_sizes.append(999)
        self.assertNotEqual(first.env_kwargs, second.env_kwargs)
        self.assertNotIn(999, second.actor_hidden_sizes)

    def test_yaml_numeric_strings_env_kwargs_and_secret_redaction(self):
        cfg = self.load(
            "n_agents: 2\nn_walkers: 2\nlearning_rate: '0.001'\ngamma: '0.5'\n"
            "env_kwargs: {max_cycles: 13}\nwandb_key: test-secret\ndevice: cpu\n"
        )
        self.assertEqual((cfg.n_agents, cfg.n_walkers), (2, 2))
        self.assertEqual((cfg.learning_rate, cfg.gamma), (0.001, 0.5))
        self.assertEqual(cfg.env_kwargs["max_cycles"], 13)
        self.assertNotIn("wandb_key", cfg.to_dict())
        self.assertNotIn("test-secret", repr(cfg.to_dict()))

    def test_counts_and_agent_walker_agreement_are_validated(self):
        for key in (
            "total_timesteps",
            "rollout_steps",
            "update_iterations",
            "minibatch_size",
            "best_window",
            "max_steps_per_episode",
            "log_interval",
            "checkpoint_interval",
            "episodes",
        ):
            for value in (0, 1.5):
                with (
                    self.subTest(key=key, value=value),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    self.load(f"{key}: {value}\n")
        with self.assertRaises(ValueError):
            self.load("n_walkers: 2\nn_agents: 3\n")

    def test_boolean_flags_and_hyperparameter_ranges_are_validated(self):
        for text in (
            "share_policy: 1\n",
            "discretize_actions: 'false'\n",
            "gamma: 1.1\n",
            "gae_lambda: -0.1\n",
            "clip_coef: 0\n",
            "learning_rate: 0\n",
            "ent_coef: -1\n",
            "target_kl: -1\n",
            "learning_rate: .nan\n",
        ):
            with self.subTest(text=text), self.assertRaises((ValueError, TypeError)):
                self.load(text)

    def test_init_checkpoint_accepts_optional_path_normalizes_and_remains_public(self):
        self.assertIsNone(Config().init_checkpoint)
        self.assertIsNone(self.load("init_checkpoint: null\n").init_checkpoint)
        raw_path = "MAPPO/checkpoints/../checkpoints_delivery/best.pt"
        cfg = self.load(f"init_checkpoint: {raw_path}\n")
        self.assertEqual(cfg.init_checkpoint, os.path.normpath(raw_path))
        self.assertEqual(cfg.to_dict()["init_checkpoint"], os.path.normpath(raw_path))
        for value in ("123", "true", "[]", "{}"):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                self.load(f"init_checkpoint: {value}\n")

    def test_mapping_and_environment_fields_are_validated(self):
        for text in (
            "- not a mapping\n",
            "env_kwargs: []\n",
            "seed: -1\n",
            "action_bins: 1\n",
            "env_kwargs: {max_cycles: 0}\n",
            "env_kwargs: {max_cycles: true}\n",
            "env_kwargs: {n_walkers: 2}\n",
            "env_kwargs: {render_mode: human}\n",
        ):
            with self.subTest(text=text), self.assertRaises((ValueError, TypeError)):
                self.load(text)


if __name__ == "__main__":
    unittest.main()
