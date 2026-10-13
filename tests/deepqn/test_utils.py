"""Checkpoint restoration includes distinct targets, Adam state, and sync schedule."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from gymnasium import spaces

from deepQN.dqn.agent import DQNAgent
import deepQN.dqn.utils as utils


def agent():
    return DQNAgent(
        spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
        spaces.Discrete(2),
        [4],
        "relu",
        0.001,
        0.5,
        3,
        1.0,
        True,
        "cpu",
    )


def batch():
    return {
        "obs": torch.tensor([[0.2, -0.1], [0.3, 0.4]]),
        "next_obs": torch.tensor([[0.3, -0.2], [0.5, 0.2]]),
        "actions": torch.tensor([0, 1]),
        "rewards": torch.tensor([1.0, -1.0]),
        "dones": torch.tensor([0.0, 1.0]),
    }


class TestCheckpoints(unittest.TestCase):
    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for one, two in zip(left, right):
                self.assert_nested_equal(one, two)
        else:
            self.assertEqual(left, right)

    def test_real_checkpoint_restores_distinct_target_adam_and_exact_next_updates(self):
        torch.manual_seed(4)
        original = agent()
        original.act(np.zeros(2, dtype=np.float32), 0.0)
        original.update(batch())
        self.assertTrue(
            any(
                not torch.equal(one, two)
                for one, two in zip(
                    original.q_network.parameters(),
                    original.target_q_network.parameters(),
                )
            )
        )
        self.assertTrue(original.optimizer.state_dict()["state"])
        expected = original.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nested", "best.pt")
            utils.save_checkpoint(
                path, original, 7, -2.0, config={"env_id": "MountainCar-v0"}
            )
            restored = agent()
            data = utils.load_checkpoint(path, restored)
        self.assertEqual((data["step"], data["best_return"]), (7, -2.0))
        self.assertEqual(data["config"], {"env_id": "MountainCar-v0"})
        self.assert_nested_equal(expected, restored.state_dict())
        # Update 2 retains the distinct target; update 3 performs hard synchronization.
        for update in (2, 3):
            left, right = original.update(batch()), restored.update(batch())
            self.assertEqual(vars(left), vars(right))
            self.assert_nested_equal(original.state_dict(), restored.state_dict())
            self.assertEqual(restored.train_steps, update)
            if update == 3:
                for online, target in zip(
                    restored.q_network.parameters(),
                    restored.target_q_network.parameters(),
                ):
                    torch.testing.assert_close(online, target, rtol=0, atol=0)

    def test_load_checkpoint_requests_weights_only_cpu_loading(self):
        original, restored = agent(), agent()
        with patch.object(
            utils.torch, "load", return_value=original.state_dict()
        ) as load:
            utils.load_checkpoint("unused.pt", restored)
        load.assert_called_once_with("unused.pt", map_location="cpu", weights_only=True)


class TestEnvironmentFactory(unittest.TestCase):
    def test_factory_seeds_reset_and_actions_and_forwards_render_kwargs(self):
        env = MagicMock()
        with (
            patch.object(utils.gym, "make", return_value=env) as make,
            patch.object(
                utils.gym.wrappers, "RecordEpisodeStatistics", return_value=env
            ),
        ):
            self.assertIs(utils.make_env("Stub-v0", 17, "rgb_array", {"g": 1.0}), env)
        make.assert_called_once_with("Stub-v0", render_mode="rgb_array", g=1.0)
        env.reset.assert_called_once_with(seed=17)
        env.action_space.seed.assert_called_once_with(17)

    def test_reset_and_action_seed_failure_close_created_environment(self):
        for location in ("reset", "seed"):
            with self.subTest(location=location):
                env = MagicMock()
                failing = env.reset if location == "reset" else env.action_space.seed
                failing.side_effect = RuntimeError("factory failure")
                with (
                    patch.object(utils.gym, "make", return_value=env),
                    patch.object(
                        utils.gym.wrappers, "RecordEpisodeStatistics", return_value=env
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, "factory failure"):
                        utils.make_env("Stub-v0", 17)
                env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
