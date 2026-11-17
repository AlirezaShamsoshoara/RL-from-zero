"""YAML configuration and secret-redaction coverage."""

import os
import tempfile
import unittest

from PPO.ppo.config import Config


class TestConfig(unittest.TestCase):
    def test_from_yaml_parses_learning_rate_and_keeps_paths_and_model_settings(
        self,
    ) -> None:
        text = (
            "env_id: Stub-v0\nseed: 19\nnum_envs: 2\n"
            "learning_rate: '0.001'\nhidden_sizes: [8, 4]\n"
            "checkpoint_dir: PPO/checkpoints_test/../checkpoints_test\n"
            "inference_model_path: PPO/checkpoints_test/best.pt\n"
            "wandb_key: config-test-secret\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "stub.yaml")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write(text)
            cfg = Config.from_yaml(path)
        self.assertEqual(cfg.env_id, "Stub-v0")
        self.assertEqual(cfg.seed, 19)
        self.assertEqual(cfg.num_envs, 2)
        self.assertEqual(cfg.learning_rate, 0.001)
        self.assertEqual(cfg.hidden_sizes, [8, 4])
        self.assertEqual(cfg.checkpoint_dir, os.path.normpath("PPO/checkpoints_test"))
        self.assertEqual(
            cfg.inference_model_path, os.path.normpath("PPO/checkpoints_test/best.pt")
        )
        self.assertEqual(cfg.wandb_key, "config-test-secret")
        self.assertNotIn("wandb_key", cfg.to_dict())
        self.assertNotIn("config-test-secret", repr(cfg.to_dict()))

    def test_from_yaml_rejects_nonpositive_count_fields(self) -> None:
        for field in (
            "num_envs",
            "total_timesteps",
            "rollout_steps",
            "update_iterations",
            "minibatch_size",
            "log_interval",
            "checkpoint_interval",
        ):
            for value in (0, -1):
                with (
                    self.subTest(field=field, value=value),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    path = os.path.join(directory, "invalid.yaml")
                    with open(path, "w", encoding="utf-8") as stream:
                        stream.write(f"{field}: {value}\n")
                    with self.assertRaisesRegex(
                        ValueError, f"{field} must be positive"
                    ):
                        Config.from_yaml(path)

    def test_from_yaml_rejects_budget_not_divisible_by_environment_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "invalid.yaml")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("num_envs: 2\ntotal_timesteps: 7\n")
            with self.assertRaisesRegex(ValueError, "divisible by num_envs"):
                Config.from_yaml(path)

    def test_from_yaml_rejects_invalid_learning_rate_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "invalid.yaml")
            with open(path, "w", encoding="utf-8") as stream:
                stream.write("learning_rate: not-a-number\n")
            with self.assertRaisesRegex(ValueError, "learning_rate must be a float"):
                Config.from_yaml(path)

    def test_default_lists_are_not_shared_between_configs(self) -> None:
        first, second = Config(), Config()
        first.hidden_sizes.append(999)
        self.assertNotIn(999, second.hidden_sizes)

    def test_to_dict_exposes_training_and_demo_settings_but_not_key(self) -> None:
        cfg = Config()
        cfg.wandb_key = "never-log-this"
        public = cfg.to_dict()
        for name in (
            "env_id",
            "seed",
            "rollout_steps",
            "log_interval",
            "checkpoint_dir",
            "inference_model_path",
            "episodes",
            "device",
        ):
            self.assertEqual(public[name], getattr(cfg, name))
        self.assertNotIn("wandb_key", public)
        self.assertNotIn("never-log-this", repr(public))


if __name__ == "__main__":
    unittest.main()
