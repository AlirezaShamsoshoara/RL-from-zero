"""Q-table checkpoint persistence and mocked environment construction."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from Qlearning.ql.agent import QLearningAgent
import Qlearning.ql.utils as ql_utils


class TestCheckpoints(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_table_counter_and_config_metadata(self):
        table = np.arange(6, dtype=np.float32).reshape(3, 2)
        metadata = {"env_id": "Stub-v0", "alpha": 0.5, "gamma": 0.9}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nested", "best.pt")
            ql_utils.save_checkpoint(
                path, table, 4, 2.5, global_step=17, config=metadata
            )
            table[:] = -999.0
            data = ql_utils.load_checkpoint(path)
        self.assertIsInstance(data["q_table"], torch.Tensor)
        self.assertEqual(data["q_table"].device.type, "cpu")
        np.testing.assert_array_equal(
            data["q_table"].numpy(), np.arange(6).reshape(3, 2)
        )
        self.assertEqual(data["step"], 4)
        self.assertEqual(data["global_step"], 17)
        self.assertEqual(data["best_return"], 2.5)
        self.assertEqual(data["config"], metadata)
        restored = QLearningAgent(3, 2, 0.5, 0.9, 1.0, 0.1, 0.01)
        restored.load_state_dict(
            {"q_table": data["q_table"], "global_step": data["global_step"]}
        )
        np.testing.assert_array_equal(restored.Q, data["q_table"].numpy())
        self.assertEqual(restored.global_step, 17)

    def test_existing_positional_checkpoint_signature_remains_usable(self):
        table = np.array([[0.0, 1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pt")
            ql_utils.save_checkpoint(path, table, 2, -1.0)
            data = ql_utils.load_checkpoint(path)
        np.testing.assert_array_equal(data["q_table"].numpy(), table)
        self.assertEqual(data["step"], 2)
        self.assertEqual(data["best_return"], -1.0)
        self.assertEqual(data.get("global_step", 0), 0)

    def test_loaded_malformed_tables_are_rejected_by_agent_state_validation(self):
        for table in (
            torch.zeros(6),
            torch.zeros((2, 2)),
            torch.full((3, 2), float("nan")),
        ):
            with self.subTest(table=table), tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "malformed.pt")
                torch.save(
                    {"q_table": table, "step": 1, "best_return": 0.0, "global_step": 0},
                    path,
                )
                agent = QLearningAgent(3, 2, 0.5, 0.9, 1.0, 0.1, 0.01)
                with self.assertRaises((ValueError, TypeError)):
                    data = ql_utils.load_checkpoint(path)
                    agent.load_state_dict(
                        {"q_table": data["q_table"], "global_step": 0}
                    )

    def test_saving_nonfinite_or_nondimensional_table_is_rejected(self):
        for table in (np.zeros(3), np.full((3, 2), np.nan), np.full((3, 2), np.inf)):
            with self.subTest(table=table), tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "bad.pt")
                with self.assertRaises(ValueError):
                    ql_utils.save_checkpoint(path, table, 1, 0.0)
                self.assertFalse(os.path.exists(path))

    def test_loaded_payload_must_be_mapping_containing_q_table(self):
        for payload in ({"step": 1}, ["not", "a", "checkpoint"]):
            with (
                self.subTest(payload=payload),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = os.path.join(directory, "bad.pt")
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, "Q-table"):
                    ql_utils.load_checkpoint(path)

    def test_legacy_table_checkpoint_is_loadable(self):
        table = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "legacy.pt")
            torch.save({"q_table": table, "step": 9, "best_return": 1.0}, path)
            data = ql_utils.load_checkpoint(path)
        agent = QLearningAgent(3, 2, 0.5, 0.9, 1.0, 0.1, 0.01)
        agent.load_state_dict(
            {"q_table": data["q_table"], "global_step": data.get("global_step", 0)}
        )
        np.testing.assert_array_equal(agent.Q, table.numpy())
        self.assertEqual(data["step"], 9)


class TestEnvironmentHelper(unittest.TestCase):
    def test_reset_failure_closes_new_environment_and_reraises(self):
        env = MagicMock()
        env.reset.side_effect = RuntimeError("stub reset failure")
        with patch.object(ql_utils.gym, "make", return_value=env):
            with self.assertRaisesRegex(RuntimeError, "stub reset failure"):
                ql_utils.make_env("Stub-v0", 19)
        env.close.assert_called_once()
        env.action_space.seed.assert_not_called()

    def test_action_seed_failure_closes_new_environment_and_reraises(self):
        env = MagicMock()
        env.action_space.seed.side_effect = RuntimeError("stub seed failure")
        with patch.object(ql_utils.gym, "make", return_value=env):
            with self.assertRaisesRegex(RuntimeError, "stub seed failure"):
                ql_utils.make_env("Stub-v0", 19)
        env.reset.assert_called_once_with(seed=19)
        env.close.assert_called_once()

    def test_make_env_forwards_kwargs_rendering_and_seed(self):
        env = MagicMock()
        with patch.object(ql_utils.gym, "make", return_value=env) as make:
            actual = ql_utils.make_env(
                "Stub-v0",
                19,
                render_mode="ansi",
                env_kwargs={"is_slippery": False, "map_name": "8x8"},
            )
        self.assertIs(actual, env)
        make.assert_called_once_with(
            "Stub-v0", render_mode="ansi", is_slippery=False, map_name="8x8"
        )
        env.reset.assert_called_once_with(seed=19)
        env.action_space.seed.assert_called_once_with(19)


if __name__ == "__main__":
    unittest.main()
