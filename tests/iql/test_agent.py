"""IQL agent tests.

Examples:
    >>> # python -m pytest tests/iql/test_agent.py
    >>> # python -m unittest tests.iql.test_agent
"""

import unittest

import numpy as np
import torch
from gymnasium import spaces

from IQL.iql.agent import IQLAgent, _expectile_loss


def _make_agent(obs_dim=3, act_dim=1, hidden_sizes=(16,)):
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-2.0, high=2.0, shape=(act_dim,), dtype=np.float32)
    return IQLAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=hidden_sizes,
        activation="relu",
        actor_lr=3e-4,
        critic_lr=3e-4,
        value_lr=3e-4,
        gamma=0.99,
        expectile=0.7,
        temperature=3.0,
        max_weight=100.0,
        tau=0.005,
        device="cpu",
    )


def _batch(batch_size=32, obs_dim=3, act_dim=1):
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.tanh(torch.randn(batch_size, act_dim)) * 2.0
    rewards = torch.randn(batch_size, 1)
    next_obs = torch.randn(batch_size, obs_dim)
    dones = torch.zeros(batch_size, 1)
    return obs, actions, rewards, next_obs, dones


class TestExpectileLoss(unittest.TestCase):
    def test_asymmetric_weighting(self) -> None:
        diff_pos = torch.tensor([1.0])
        diff_neg = torch.tensor([-1.0])
        # expectile 0.7 weights positive diffs more than negative ones.
        self.assertAlmostEqual(float(_expectile_loss(diff_pos, 0.7)), 0.7, places=5)
        self.assertAlmostEqual(float(_expectile_loss(diff_neg, 0.7)), 0.3, places=5)


class TestIQLAgent(unittest.TestCase):
    def test_act_in_bounds(self) -> None:
        agent = _make_agent()
        obs = np.zeros(3, dtype=np.float32)
        det = agent.act(obs, deterministic=True)
        sto = agent.act(obs, deterministic=False)
        self.assertEqual(det.shape, (1,))
        self.assertEqual(sto.shape, (1,))
        self.assertTrue(np.all(det <= 2.0 + 1e-4) and np.all(det >= -2.0 - 1e-4))
        self.assertTrue(np.all(sto <= 2.0 + 1e-4) and np.all(sto >= -2.0 - 1e-4))

    def test_update_returns_finite_and_changes_params(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        agent = _make_agent(hidden_sizes=(32,))
        actor_before = [p.detach().clone() for p in agent.actor.parameters()]
        value_target_before = [p.detach().clone() for p in agent.value_target.parameters()]

        stats = agent.update(_batch())

        self.assertEqual(agent.global_step, 1)
        for v in (stats.critic_loss, stats.value_loss, stats.actor_loss,
                  stats.mean_advantage, stats.weight_mean, stats.weight_max):
            self.assertTrue(np.isfinite(v))
        # actor and the soft-updated value target should have moved.
        self.assertTrue(any(
            not torch.allclose(b, a)
            for b, a in zip(actor_before, agent.actor.parameters())
        ))
        self.assertTrue(any(
            not torch.allclose(b, a)
            for b, a in zip(value_target_before, agent.value_target.parameters())
        ))

    def test_weight_clipping(self) -> None:
        agent = _make_agent()
        agent.max_weight = 5.0
        stats = agent.update(_batch())
        self.assertLessEqual(stats.weight_max, 5.0 + 1e-4)

    def test_state_dict_round_trip(self) -> None:
        torch.manual_seed(1)
        agent = _make_agent()
        agent.update(_batch())
        restored = _make_agent()
        restored.load_state_dict(agent.state_dict())
        self.assertEqual(restored.global_step, agent.global_step)
        for p, q in zip(agent.actor.parameters(), restored.actor.parameters()):
            self.assertTrue(torch.allclose(p, q))
        for p, q in zip(agent.q1.parameters(), restored.q1.parameters()):
            self.assertTrue(torch.allclose(p, q))
        for p, q in zip(agent.value.parameters(), restored.value.parameters()):
            self.assertTrue(torch.allclose(p, q))


if __name__ == "__main__":
    unittest.main()
