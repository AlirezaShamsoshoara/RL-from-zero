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
        obs = np.arange(batch * self.obs_dim, dtype=np.float32).reshape(
            batch, self.obs_dim
        )
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


class TestReplayContracts(unittest.TestCase):
    def _buffer(self, capacity=5):
        return ReplayBuffer(3, 2, capacity, torch.device("cpu"))

    def _batch(self, n, offset=0):
        values = np.arange(offset, offset + n, dtype=np.float32)
        obs = np.repeat(values[:, None], 3, axis=1)
        acts = np.repeat((values / 100)[:, None], 2, axis=1)
        return obs, acts, values, obs + 10, (values.astype(int) % 2).astype(np.float32)

    def _snapshot(self, buffer):
        return (
            buffer.ptr,
            buffer.size,
            [
                getattr(buffer, name).copy()
                for name in [
                    "obs_buf",
                    "acts_buf",
                    "rews_buf",
                    "next_obs_buf",
                    "done_buf",
                ]
            ],
        )

    def _assert_snapshot(self, buffer, snapshot):
        self.assertEqual((buffer.ptr, buffer.size), snapshot[:2])
        for current, old in zip(self._snapshot(buffer)[2], snapshot[2]):
            np.testing.assert_array_equal(current, old)

    def test_constructor_counts_are_positive_strict_integers(self):
        for name in ["obs_dim", "act_dim", "capacity"]:
            for value in [0, -1, 1.5, True]:
                args = dict(
                    obs_dim=3, act_dim=2, capacity=5, device=torch.device("cpu")
                )
                args[name] = value
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    ReplayBuffer(**args)

    def test_batch_larger_than_capacity_matches_full_sequential_storage(self):
        for prefix in [0, 2, 4, 7]:
            for n in [5, 6, 11, 23]:
                with self.subTest(prefix=prefix, n=n):
                    actual, expected = self._buffer(), self._buffer()
                    for row in zip(*self._batch(prefix)):
                        actual.add(*row)
                        expected.add(*row)
                    batch = self._batch(n, offset=prefix)
                    actual.add_batch(*batch)
                    for row in zip(*batch):
                        expected.add(*row)
                    self._assert_snapshot(actual, self._snapshot(expected))

    def test_multiple_batch_wraps_match_single_insertions(self):
        actual, expected = self._buffer(), self._buffer()
        offset = 0
        for n in [3, 1, 9, 2, 17, 5]:
            batch = self._batch(n, offset)
            actual.add_batch(*batch)
            for row in zip(*batch):
                expected.add(*row)
            self._assert_snapshot(actual, self._snapshot(expected))
            offset += n

    def test_scalar_insert_validation_is_atomic_for_every_field(self):
        buffer = self._buffer()
        buffer.add_batch(*self._batch(3))
        before = self._snapshot(buffer)
        valid = [
            part[0].copy() if isinstance(part[0], np.ndarray) else part[0]
            for part in self._batch(1)
        ]
        for index in range(5):
            bad = list(valid)
            bad[index] = np.full_like(bad[index], np.nan, dtype=np.float32)
            with self.subTest(index=index), self.assertRaises((ValueError, TypeError)):
                buffer.add(*bad)
            self._assert_snapshot(buffer, before)
        for index, value in [
            (0, np.zeros(2)),
            (1, np.zeros(3)),
            (3, np.zeros((1, 3))),
            (4, 0.5),
        ]:
            bad = list(valid)
            bad[index] = value
            with self.subTest(index=index), self.assertRaises((ValueError, TypeError)):
                buffer.add(*bad)
            self._assert_snapshot(buffer, before)

    def test_batch_validation_is_atomic_after_valid_prefix_and_wrap(self):
        buffer = self._buffer()
        buffer.add_batch(*self._batch(4))
        before = self._snapshot(buffer)
        for index in range(5):
            bad = list(self._batch(7))
            bad[index][-1] = np.nan
            with self.subTest(index=index), self.assertRaises((ValueError, TypeError)):
                buffer.add_batch(*bad)
            self._assert_snapshot(buffer, before)
        for index, value in [
            (1, np.zeros((7, 3))),
            (2, np.zeros((7, 2))),
            (3, np.zeros((6, 3))),
            (4, np.full(7, 0.5)),
        ]:
            bad = list(self._batch(7))
            bad[index] = value
            with self.subTest(index=index), self.assertRaises((ValueError, TypeError)):
                buffer.add_batch(*bad)
            self._assert_snapshot(buffer, before)

    def test_insertions_copy_caller_arrays_and_column_scalars(self):
        buffer = self._buffer()
        batch = list(self._batch(3))
        batch[2] = batch[2][:, None]
        batch[4] = batch[4][:, None]
        buffer.add_batch(*batch)
        before = self._snapshot(buffer)
        for field in batch:
            field[:] = 99
        self._assert_snapshot(buffer, before)

    def test_samples_copy_storage_and_each_other_with_replacement(self):
        from unittest.mock import patch

        buffer = self._buffer()
        buffer.add_batch(*self._batch(5))
        before = self._snapshot(buffer)
        with patch("numpy.random.randint", return_value=np.array([2, 2, 2])):
            first = buffer.sample(3)
            second = buffer.sample(3)
        for tensor in first:
            tensor.fill_(999)
        self._assert_snapshot(buffer, before)
        expected = [
            buffer.obs_buf,
            buffer.acts_buf,
            buffer.rews_buf,
            buffer.next_obs_buf,
            buffer.done_buf,
        ]
        for tensor, array in zip(second, expected):
            np.testing.assert_array_equal(tensor.numpy(), array[[2, 2, 2]])
            self.assertEqual(tensor.dtype, torch.float32)

    def test_sample_counts_and_empty_buffer_fail_without_storage_changes(self):
        buffer = self._buffer()
        before = self._snapshot(buffer)
        with self.assertRaises((ValueError, RuntimeError)):
            buffer.sample(1)
        buffer.add_batch(*self._batch(3))
        before = self._snapshot(buffer)
        for n in [0, -1, 1.5, True]:
            for operation in [buffer.sample, buffer.can_sample]:
                with (
                    self.subTest(n=n, operation=operation),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    operation(n)
            self._assert_snapshot(buffer, before)


if __name__ == "__main__":
    unittest.main()
