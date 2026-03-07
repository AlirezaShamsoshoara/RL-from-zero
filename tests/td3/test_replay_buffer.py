"""TD3 replay buffer tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/td3/test_replay_buffer.py
    >>> # Run with unittest
    >>> # python -m unittest tests.td3.test_replay_buffer
"""

import unittest

import numpy as np
import torch

from TD3.td3.replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def setUp(self) -> None:
        self.obs_dim = 3
        self.act_dim = 2
        self.capacity = 5
        self.device = torch.device("cpu")
        self.buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            capacity=self.capacity,
            device=self.device,
        )

    def test_add_increments_size(self) -> None:
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        next_obs = np.array([3.0, 2.0, 1.0], dtype=np.float32)
        action = np.array([0.5, -0.5], dtype=np.float32)
        reward = 1.0
        done = 0.0

        self.buffer.add(obs, action, reward, next_obs, done)

        self.assertEqual(self.buffer.size, 1)
        np.testing.assert_array_equal(self.buffer.obs_buf[0], obs)
        np.testing.assert_array_equal(self.buffer.next_obs_buf[0], next_obs)
        np.testing.assert_array_equal(self.buffer.acts_buf[0], action)
        self.assertAlmostEqual(float(self.buffer.rews_buf[0, 0]), reward)
        self.assertAlmostEqual(float(self.buffer.done_buf[0, 0]), done)

    def test_overwrites_oldest_when_full(self) -> None:
        for idx in range(self.capacity + 2):
            obs = np.full(self.obs_dim, float(idx), dtype=np.float32)
            next_obs = obs + 1.0
            action = np.full(self.act_dim, float(idx), dtype=np.float32)
            self.buffer.add(obs, action, float(idx), next_obs, 0.0)

        self.assertEqual(self.buffer.size, self.capacity)
        expected_values = {float(idx) for idx in range(2, self.capacity + 2)}
        stored_values = {float(row[0]) for row in self.buffer.obs_buf}
        self.assertSetEqual(stored_values, expected_values)

    def test_can_sample_and_sample_shapes(self) -> None:
        for idx in range(self.capacity):
            obs = np.full(self.obs_dim, float(idx), dtype=np.float32)
            next_obs = obs + 1.0
            action = np.full(self.act_dim, float(idx), dtype=np.float32)
            self.buffer.add(obs, action, float(idx), next_obs, 0.0)

        self.assertTrue(self.buffer.can_sample(self.capacity))
        self.assertFalse(self.buffer.can_sample(self.capacity + 1))

        batch_size = 3
        np.random.seed(0)
        obs, acts, rews, next_obs, dones = self.buffer.sample(batch_size)

        self.assertEqual(obs.shape, (batch_size, self.obs_dim))
        self.assertEqual(acts.shape, (batch_size, self.act_dim))
        self.assertEqual(rews.shape, (batch_size, 1))
        self.assertEqual(next_obs.shape, (batch_size, self.obs_dim))
        self.assertEqual(dones.shape, (batch_size, 1))
        self.assertEqual(obs.dtype, torch.float32)
        self.assertEqual(acts.dtype, torch.float32)
        self.assertEqual(rews.dtype, torch.float32)
        self.assertEqual(obs.device.type, self.device.type)

    def test_add_batch_inserts_multiple_transitions(self) -> None:
        batch = 3
        obs = np.arange(batch * self.obs_dim, dtype=np.float32).reshape(batch, self.obs_dim)
        acts = np.ones((batch, self.act_dim), dtype=np.float32)
        rews = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        next_obs = obs + 10.0
        dones = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.buffer.add_batch(obs, acts, rews, next_obs, dones)

        self.assertEqual(self.buffer.size, batch)
        self.assertEqual(self.buffer.ptr, batch)
        np.testing.assert_array_equal(self.buffer.obs_buf[:batch], obs)
        np.testing.assert_array_equal(self.buffer.next_obs_buf[:batch], next_obs)
        np.testing.assert_array_almost_equal(
            self.buffer.rews_buf[:batch].flatten(), rews
        )
        np.testing.assert_array_almost_equal(
            self.buffer.done_buf[:batch].flatten(), dones
        )

    def test_add_batch_wraps_around(self) -> None:
        # Fill buffer to near capacity, then add a batch that wraps
        for idx in range(self.capacity - 1):
            obs = np.full(self.obs_dim, float(idx), dtype=np.float32)
            self.buffer.add(obs, np.zeros(self.act_dim), 0.0, obs, 0.0)

        self.assertEqual(self.buffer.ptr, self.capacity - 1)

        batch = 3
        obs = np.full((batch, self.obs_dim), 99.0, dtype=np.float32)
        acts = np.zeros((batch, self.act_dim), dtype=np.float32)
        rews = np.ones(batch, dtype=np.float32)
        next_obs = obs.copy()
        dones = np.zeros(batch, dtype=np.float32)

        self.buffer.add_batch(obs, acts, rews, next_obs, dones)

        self.assertEqual(self.buffer.size, self.capacity)
        # ptr should wrap: (capacity-1 + 3) % capacity = 2
        self.assertEqual(self.buffer.ptr, 2)
        # Last slot and first two slots should contain the batch data
        np.testing.assert_array_equal(self.buffer.obs_buf[self.capacity - 1], obs[0])
        np.testing.assert_array_equal(self.buffer.obs_buf[0], obs[1])
        np.testing.assert_array_equal(self.buffer.obs_buf[1], obs[2])


if __name__ == "__main__":
    unittest.main()
