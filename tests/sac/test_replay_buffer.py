"""SAC replay buffer tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/sac/test_replay_buffer.py
    >>> # Run with unittest
    >>> # python -m unittest tests.sac.test_replay_buffer
"""

import unittest

import numpy as np
import torch

from SAC.sac.replay_buffer import ReplayBuffer


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

    def test_dimensions_and_capacity_require_positive_integer_counts(self) -> None:
        for field in ["obs_dim", "act_dim", "capacity"]:
            for value in [0, -1, True, 1.5]:
                kwargs = dict(obs_dim=3, act_dim=2, capacity=5, device=self.device)
                kwargs[field] = value
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    ReplayBuffer(**kwargs)

    def test_batch_counts_require_positive_integers_for_both_entry_points(self) -> None:
        self._add_valid()
        for value in [0, -1, True, 1.5]:
            for method in [self.buffer.can_sample, self.buffer.sample]:
                with (
                    self.subTest(value=value, method=method.__name__),
                    self.assertRaises(ValueError),
                ):
                    method(value)

    def test_empty_sample_rejected_but_nonempty_sample_is_with_replacement(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            self.buffer.sample(1)
        self._add_valid()
        self.assertFalse(self.buffer.can_sample(6))
        sampled = self.buffer.sample(6)
        self.assertEqual(sampled[0].shape, (6, 3))
        torch.testing.assert_close(
            sampled[0], torch.tensor([[1.0, 2.0, 3.0]]).expand(6, 3)
        )

    def test_add_requires_exact_vector_shapes_without_partial_mutation(self) -> None:
        invalid = [
            (0, np.zeros(1)),
            (0, np.zeros((1, 3))),
            (1, np.zeros(1)),
            (1, np.zeros((1, 2))),
            (3, np.zeros(2)),
        ]
        self._assert_invalid_adds(invalid)

    def test_nonfinite_vectors_and_float32_cast_overflow_do_not_mutate(self) -> None:
        invalid = [
            (0, np.array([1.0, np.nan, 3.0])),
            (1, np.array([np.inf, 0.0])),
            (3, np.array([0.0, -np.inf, 0.0])),
            (0, np.array([1e100, 0.0, 0.0])),
        ]
        self._assert_invalid_adds(invalid)

    def test_reward_requires_a_finite_float32_representable_scalar(self) -> None:
        self._assert_invalid_adds(
            [
                (2, value)
                for value in [
                    float("nan"),
                    float("inf"),
                    1e100,
                    np.array([1.0]),
                    np.array([[1.0]]),
                ]
            ]
        )

    def test_done_is_a_scalar_bitmask_and_invalid_bits_do_not_mutate(self) -> None:
        self._assert_invalid_adds(
            [(4, value) for value in [-1.0, 0.5, 2.0, float("nan"), np.array([1.0])]]
        )
        for value in [False, True, 0.0, 1.0]:
            args = self._transition()
            args[4] = value
            self.buffer.add(*args)
        self.assertEqual(self.buffer.size, 5)
        self.assertTrue(set(self.buffer.done_buf[:, 0]).issubset({0.0, 1.0}))

    def test_add_owns_caller_arrays_and_accepts_finite_zero_dimensional_scalars(
        self,
    ) -> None:
        args = self._transition()
        args[2] = np.array(1.0)
        args[4] = np.array(0.0)
        self.buffer.add(*args)
        expected = self._snapshot()
        for index in [0, 1, 3]:
            args[index].fill(-99.0)
        self._assert_snapshot(expected)

    def test_sample_returns_owned_cpu_copies_independent_of_future_writes(self) -> None:
        self._add_valid()
        before = self._snapshot()
        batch = self.buffer.sample(1)
        for tensor in batch:
            self.assertEqual(tensor.device.type, "cpu")
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertFalse(tensor.requires_grad)
            tensor.fill_(-99.0)
        self._assert_snapshot(before)
        second = self.buffer.sample(1)
        saved = tuple(t.clone() for t in second)
        for _ in range(self.capacity):
            self._add_valid(offset=5.0)
        for tensor, original in zip(second, saved):
            torch.testing.assert_close(tensor, original)

    def test_seeded_sampling_is_reproducible_and_preserves_transition_alignment(
        self,
    ) -> None:
        for offset in range(self.capacity):
            self._add_valid(offset=float(offset))
        saved_rng = np.random.get_state()
        try:
            np.random.seed(12)
            first = self.buffer.sample(4)
            np.random.seed(12)
            second = self.buffer.sample(4)
        finally:
            np.random.set_state(saved_rng)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
        obs, actions, rewards, next_obs, dones = first
        torch.testing.assert_close(next_obs, obs + 1.0)
        torch.testing.assert_close(rewards[:, 0], obs[:, 0])
        torch.testing.assert_close(actions[:, 0], obs[:, 0] - 0.5)
        torch.testing.assert_close(dones, torch.zeros_like(dones))

    @staticmethod
    def _transition(offset=0.0):
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32) + offset
        action = np.array([0.5, -0.5], dtype=np.float32) + offset
        return [obs, action, float(1.0 + offset), obs + 1.0, 0.0]

    def _add_valid(self, offset=0.0):
        self.buffer.add(*self._transition(offset))

    def _snapshot(self):
        arrays = [
            array.copy()
            for array in [
                self.buffer.obs_buf,
                self.buffer.acts_buf,
                self.buffer.rews_buf,
                self.buffer.next_obs_buf,
                self.buffer.done_buf,
            ]
        ]
        return self.buffer.ptr, self.buffer.size, arrays

    def _assert_snapshot(self, expected):
        ptr, size, arrays = expected
        self.assertEqual(self.buffer.ptr, ptr)
        self.assertEqual(self.buffer.size, size)
        actual = [
            self.buffer.obs_buf,
            self.buffer.acts_buf,
            self.buffer.rews_buf,
            self.buffer.next_obs_buf,
            self.buffer.done_buf,
        ]
        for left, right in zip(actual, arrays):
            np.testing.assert_array_equal(left, right)

    def _assert_invalid_adds(self, invalid):
        self._add_valid()
        before = self._snapshot()
        for index, value in invalid:
            args = self._transition(offset=10.0)
            args[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                self.buffer.add(*args)
            self._assert_snapshot(before)


if __name__ == "__main__":
    unittest.main()
