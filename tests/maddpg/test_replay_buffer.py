"""MADDPG multi-agent replay buffer tests.

Examples:
    >>> # python -m pytest tests/maddpg/test_replay_buffer.py
"""

import unittest

import numpy as np
import torch

from MADDPG.maddpg.replay_buffer import MultiAgentReplayBuffer


def _add_random(buffer: MultiAgentReplayBuffer, n: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        obs = [rng.standard_normal(d).astype(np.float32) for d in buffer.obs_dims]
        acts = [rng.uniform(-1.0, 1.0, d).astype(np.float32) for d in buffer.act_dims]
        rews = [float(rng.standard_normal()) for _ in range(buffer.n_agents)]
        next_obs = [rng.standard_normal(d).astype(np.float32) for d in buffer.obs_dims]
        dones = [bool(rng.integers(0, 2)) for _ in range(buffer.n_agents)]
        buffer.add(obs, acts, rews, next_obs, dones)


class TestMultiAgentReplayBuffer(unittest.TestCase):
    def test_can_sample_only_after_enough_data(self) -> None:
        buf = MultiAgentReplayBuffer(
            n_agents=2, obs_dims=[3, 4], act_dims=[1, 2],
            capacity=100, device=torch.device("cpu"),
        )
        self.assertFalse(buf.can_sample(8))
        _add_random(buf, 8)
        self.assertTrue(buf.can_sample(8))

    def test_sample_shapes_per_agent(self) -> None:
        buf = MultiAgentReplayBuffer(
            n_agents=3, obs_dims=[5, 4, 6], act_dims=[2, 3, 1],
            capacity=64, device=torch.device("cpu"),
        )
        _add_random(buf, 32)
        obs, acts, rews, next_obs, dones = buf.sample(16)
        self.assertEqual(len(obs), 3)
        for i, (o, a, r, n, d) in enumerate(zip(obs, acts, rews, next_obs, dones)):
            self.assertEqual(o.shape, (16, buf.obs_dims[i]))
            self.assertEqual(a.shape, (16, buf.act_dims[i]))
            self.assertEqual(r.shape, (16, 1))
            self.assertEqual(n.shape, (16, buf.obs_dims[i]))
            self.assertEqual(d.shape, (16, 1))

    def test_circular_overwrite(self) -> None:
        buf = MultiAgentReplayBuffer(
            n_agents=2, obs_dims=[2, 2], act_dims=[1, 1],
            capacity=4, device=torch.device("cpu"),
        )
        _add_random(buf, 10)
        self.assertEqual(buf.size, 4)  # never exceeds capacity
        # After wrap-around all stored transitions must still be finite.
        obs, *_ = buf.sample(4)
        self.assertTrue(torch.isfinite(obs[0]).all().item())

    def test_joint_alignment_preserved(self) -> None:
        # Store a deterministic pattern; verify that a sample's obs[i], act[i], etc.,
        # come from the same time step across agents.
        buf = MultiAgentReplayBuffer(
            n_agents=2, obs_dims=[1, 1], act_dims=[1, 1],
            capacity=10, device=torch.device("cpu"),
        )
        for t in range(10):
            buf.add(
                obs_list=[np.array([float(t)], np.float32), np.array([-float(t)], np.float32)],
                acts_list=[np.array([float(t)], np.float32), np.array([-float(t)], np.float32)],
                rewards=[float(t), -float(t)],
                next_obs_list=[np.array([float(t + 1)], np.float32),
                               np.array([-float(t + 1)], np.float32)],
                dones=[False, False],
            )
        obs, acts, rews, next_obs, dones = buf.sample(4)
        # For every sampled index, agent-0 obs is +t and agent-1 obs is -t.
        self.assertTrue(torch.allclose(obs[0], -obs[1]))
        self.assertTrue(torch.allclose(acts[0], -acts[1]))
        self.assertTrue(torch.allclose(rews[0], -rews[1]))


if __name__ == "__main__":
    unittest.main()
