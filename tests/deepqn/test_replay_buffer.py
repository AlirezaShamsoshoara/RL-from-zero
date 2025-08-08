import unittest

import numpy as np
import torch

from deepQN.dqn.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def setUp(self) -> None:
        self.obs_dim = 4
        self.capacity = 5
        self.device = torch.device("cpu")
        self.buffer = ReplayBuffer(self.obs_dim, self.capacity, self.device)

    def test_add_increments_length_and_stores_values(self) -> None:
        obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        next_obs = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        action = 2
        reward = 1.5
        done = 0.0

        self.buffer.add(obs, action, reward, next_obs, done)

        self.assertEqual(len(self.buffer), 1)
        np.testing.assert_array_equal(self.buffer.obs[0], obs)
        np.testing.assert_array_equal(self.buffer.next_obs[0], next_obs)
        self.assertEqual(self.buffer.actions[0], action)
        self.assertAlmostEqual(self.buffer.rewards[0], reward)
        self.assertAlmostEqual(self.buffer.dones[0], done)

    def test_overwrites_oldest_transition_when_full(self) -> None:
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
        with self.assertRaises(ValueError):
            self.buffer.sample(1)
        self.assertFalse(self.buffer.can_sample(1))


if __name__ == "__main__":
    # Run via `python -m unittest tests.deepqn.test_replay_buffer`
    unittest.main()
