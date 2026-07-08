"""IQL offline-dataset tests.

Examples:
    >>> # python -m pytest tests/iql/test_dataset.py
    >>> # python -m unittest tests.iql.test_dataset
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from IQL.iql.dataset import OfflineDataset, build_dataset


def _arrays(n=50, obs_dim=3, act_dim=1):
    obs = np.random.randn(n, obs_dim).astype(np.float32)
    actions = np.random.randn(n, act_dim).astype(np.float32)
    rewards = np.random.randn(n).astype(np.float32)
    next_obs = np.random.randn(n, obs_dim).astype(np.float32)
    dones = (np.random.rand(n) < 0.1).astype(np.float32)
    return obs, actions, rewards, next_obs, dones


class TestOfflineDataset(unittest.TestCase):
    def test_from_arrays_and_sample(self) -> None:
        obs, actions, rewards, next_obs, dones = _arrays(n=40)
        ds = OfflineDataset.from_arrays(obs, actions, rewards, next_obs, dones,
                                        device=torch.device("cpu"))
        self.assertEqual(ds.size, 40)
        self.assertEqual(ds.obs_dim, 3)
        self.assertEqual(ds.act_dim, 1)

        b_obs, b_act, b_rew, b_next, b_done = ds.sample(8)
        self.assertEqual(b_obs.shape, (8, 3))
        self.assertEqual(b_act.shape, (8, 1))
        self.assertEqual(b_rew.shape, (8, 1))
        self.assertEqual(b_next.shape, (8, 3))
        self.assertEqual(b_done.shape, (8, 1))
        self.assertEqual(b_obs.dtype, torch.float32)

    def test_sample_rejects_nonpositive(self) -> None:
        ds = OfflineDataset.from_arrays(*_arrays(n=10), device=torch.device("cpu"))
        with self.assertRaises(ValueError):
            ds.sample(0)


class TestBuildDataset(unittest.TestCase):
    def test_random_source_shapes_and_stats(self) -> None:
        ds, stats = build_dataset(
            source="random", env_id="Pendulum-v1", seed=0,
            device=torch.device("cpu"), num_steps=200, env_kwargs={},
        )
        self.assertEqual(stats.size, 200)
        self.assertEqual(ds.size, 200)
        self.assertEqual(ds.obs_dim, 3)
        self.assertEqual(ds.act_dim, 1)

    def test_reward_transforms(self) -> None:
        ds, stats = build_dataset(
            source="random", env_id="Pendulum-v1", seed=0,
            device=torch.device("cpu"), num_steps=200, env_kwargs={},
            normalize_rewards=True,
        )
        # After normalization the reward mean is ~0.
        self.assertAlmostEqual(stats.reward_mean, 0.0, places=4)

    def test_npz_source_round_trip(self) -> None:
        obs, actions, rewards, next_obs, dones = _arrays(n=30)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.npz")
            np.savez(path, observations=obs, actions=actions, rewards=rewards,
                     next_observations=next_obs, terminals=dones)
            ds, stats = build_dataset(
                source="npz", env_id="Pendulum-v1", seed=0,
                device=torch.device("cpu"), num_steps=0, path=path,
            )
            self.assertEqual(stats.size, 30)

    def test_mixed_source_higher_quality_than_random(self) -> None:
        # The mixed (scripted + noise) dataset should have a higher mean reward
        # than the random dataset on Pendulum.
        rnd, rnd_stats = build_dataset(
            source="random", env_id="Pendulum-v1", seed=0,
            device=torch.device("cpu"), num_steps=3000, env_kwargs={},
        )
        mix, mix_stats = build_dataset(
            source="mixed", env_id="Pendulum-v1", seed=0,
            device=torch.device("cpu"), num_steps=3000, env_kwargs={},
        )
        self.assertEqual(mix_stats.size, 3000)
        self.assertGreater(mix_stats.reward_mean, rnd_stats.reward_mean)

    def test_unsupported_source_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_dataset(source="bogus", env_id="Pendulum-v1", seed=0,
                          device=torch.device("cpu"), num_steps=10)


if __name__ == "__main__":
    unittest.main()
