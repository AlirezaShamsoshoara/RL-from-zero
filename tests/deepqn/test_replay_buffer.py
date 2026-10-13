import unittest

import numpy as np
import torch
from gymnasium import spaces

from deepQN.dqn.agent import DQNAgent
from deepQN.dqn.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def setUp(self) -> None:
        """Instantiate a fresh replay buffer on CPU for each test case."""
        self.obs_dim = 4
        self.capacity = 5
        self.device = torch.device("cpu")
        self.buffer = ReplayBuffer(self.obs_dim, self.capacity, self.device)

    def test_add_increments_length_and_stores_values(self) -> None:
        """Ensure pushed transitions are stored verbatim and the buffer length updates."""
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        next_obs = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        action, reward, done = 2, 1.5, 0.0
        self.buffer.add(obs, action, reward, next_obs, done)
        self.assertEqual(len(self.buffer), 1)
        np.testing.assert_array_equal(self.buffer.obs[0], obs)
        np.testing.assert_array_equal(self.buffer.next_obs[0], next_obs)
        self.assertEqual(self.buffer.actions[0], action)
        self.assertAlmostEqual(self.buffer.rewards[0], reward)
        self.assertAlmostEqual(self.buffer.dones[0], done)

    def test_overwrites_oldest_transition_when_full(self) -> None:
        """Verify circular buffering overwrites the oldest entries once capacity is exceeded."""
        for idx in range(self.capacity + 2):
            value = float(idx)
            obs = np.full(self.obs_dim, value, dtype=np.float32)
            next_obs = np.full(self.obs_dim, value + 0.5, dtype=np.float32)
            self.buffer.add(obs, idx, value, next_obs, float(idx % 2))
        self.assertEqual(len(self.buffer), self.capacity)
        expected_values = {float(idx) for idx in range(2, self.capacity + 2)}
        stored_values = {float(row[0]) for row in self.buffer.obs}
        self.assertSetEqual(stored_values, expected_values)

    def test_can_sample_and_sample_shapes(self) -> None:
        """Check sampling preconditions and that returned tensors have expected shapes and dtypes."""
        for idx in range(self.capacity):
            vector = np.full(self.obs_dim, float(idx), dtype=np.float32)
            self.buffer.add(vector, idx, float(idx), vector + 1.0, 0.0)
        self.assertTrue(self.buffer.can_sample(self.capacity))
        self.assertFalse(self.buffer.can_sample(self.capacity + 1))
        batch_size = 3
        np.random.seed(0)
        batch = self.buffer.sample(batch_size)
        self.assertEqual(batch["obs"].shape, (batch_size, self.obs_dim))
        self.assertEqual(batch["next_obs"].shape, (batch_size, self.obs_dim))
        self.assertEqual(batch["obs"].dtype, torch.float32)
        self.assertEqual(batch["rewards"].dtype, torch.float32)
        self.assertEqual(batch["actions"].dtype, torch.int64)
        self.assertEqual(batch["dones"].dtype, torch.float32)
        self.assertEqual(batch["obs"].device.type, self.device.type)

    def test_sample_raises_with_insufficient_data(self) -> None:
        """Assert sampling without enough stored transitions raises a ValueError."""
        with self.assertRaises(ValueError):
            self.buffer.sample(1)
        self.assertFalse(self.buffer.can_sample(1))


class TestReplayContracts(unittest.TestCase):
    def test_invalid_dimensions_and_batches_are_rejected(self):
        for obs_dim, capacity in ((0, 3), (2, 0), (-1, 3), (1.5, 3)):
            with (
                self.subTest(obs_dim=obs_dim, capacity=capacity),
                self.assertRaises((ValueError, TypeError)),
            ):
                ReplayBuffer(obs_dim, capacity, torch.device("cpu"))
        buffer = ReplayBuffer(2, 3, torch.device("cpu"))
        buffer.add(np.zeros(2), 0, 1.0, np.ones(2), 0.0)
        for size in (0, -1, 1.5):
            with self.subTest(size=size), self.assertRaises((ValueError, TypeError)):
                buffer.sample(size)

    def test_raw_obs_storage_copies_inputs_and_preserves_native_negative_reward(self):
        buffer = ReplayBuffer(2, 3, torch.device("cpu"))
        obs, next_obs = np.array([-0.5, 0.01]), np.array([-0.4, 0.02])
        buffer.add(obs, 2, -1.0, next_obs, 0.0)
        obs[:] = 99.0
        next_obs[:] = 99.0
        np.testing.assert_allclose(buffer.obs[0], [-0.5, 0.01])
        np.testing.assert_allclose(buffer.next_obs[0], [-0.4, 0.02])
        self.assertEqual(buffer.rewards[0], -1.0)
        self.assertEqual(buffer.dones[0], 0.0)

    def test_normalized_agent_update_does_not_encode_stored_replay_or_shape_reward(
        self,
    ):
        low, high = (
            np.array([-2.0, 0.0], dtype=np.float32),
            np.array([2.0, 8.0], dtype=np.float32),
        )
        agent = DQNAgent(
            spaces.Box(low, high, dtype=np.float32),
            spaces.Discrete(2),
            [],
            "relu",
            0.001,
            0.5,
            100,
            1.0,
            False,
            "cpu",
            normalize_observations=True,
        )
        buffer = ReplayBuffer(2, 3, torch.device("cpu"))
        buffer.add(np.array([2.0, 5.0]), 0, -1.0, np.array([0.0, 8.0]), 0.0)
        agent.update(buffer.sample(1))
        np.testing.assert_array_equal(buffer.obs[0], [2.0, 5.0])
        np.testing.assert_array_equal(buffer.next_obs[0], [0.0, 8.0])
        self.assertEqual(buffer.rewards[0], -1.0)
        self.assertEqual(buffer.dones[0], 0.0)

    def test_nonfinite_shape_action_and_nonbinary_done_inputs_are_rejected(self):
        for key, value in (
            ("obs", [np.nan, 0.0]),
            ("next_obs", [np.inf, 0.0]),
            ("reward", np.nan),
            ("done", 0.5),
            ("done", -1.0),
            ("obs", [1.0]),
            ("action", -1),
            ("action", 1.5),
        ):
            with self.subTest(key=key, value=value):
                buffer = ReplayBuffer(2, 3, torch.device("cpu"))
                args = dict(
                    obs=np.zeros(2), action=0, reward=1.0, next_obs=np.ones(2), done=0.0
                )
                args[key] = value
                with self.assertRaises((ValueError, TypeError)):
                    buffer.add(**args)
                self.assertEqual(len(buffer), 0)
                self.assertEqual(buffer.ptr, 0)


if __name__ == "__main__":
    unittest.main()
