"""TRPO agent tests (continuous and discrete action spaces).

Examples:
    >>> # python -m pytest tests/trpo/test_agent.py
    >>> # python -m unittest tests.trpo.test_agent
"""

import unittest

import numpy as np
import torch
from gymnasium import spaces

from TRPO.trpo.agent import Batch, TRPOAgent


def _make_agent(act_space, hidden_sizes=(16,), max_kl=0.01):
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
    return TRPOAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=hidden_sizes,
        activation="tanh",
        max_kl=max_kl,
        cg_iters=5,
        cg_damping=0.1,
        line_search_coef=0.8,
        line_search_steps=10,
        vf_lr=1e-3,
        vf_iters=2,
        entropy_coef=0.0,
        normalize_advantages=True,
        device="cpu",
    )


def _batch(agent, batch_size=24):
    obs = torch.randn(batch_size, 5)
    if agent.discrete:
        actions = torch.randint(0, agent.num_actions, (batch_size,))
    else:
        actions = torch.tanh(torch.randn(batch_size, agent.act_dim))
    logprobs = agent.policy.log_prob(obs, actions).detach()
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)
    return Batch(obs=obs, actions=actions, logprobs=logprobs,
                 returns=returns, advantages=advantages)


class TestTRPOAgentContinuous(unittest.TestCase):
    def setUp(self) -> None:
        self.act_space = spaces.Box(low=-2.0, high=2.0, shape=(2,), dtype=np.float32)

    def test_act_shapes_dtype_and_bounds(self) -> None:
        agent = _make_agent(self.act_space)
        obs = np.zeros(5, dtype=np.float32)
        a, lp, v = agent.act(obs)
        ad, _, _ = agent.act(obs, deterministic=True)
        self.assertEqual(a.shape, (1, 2))
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(np.all(a <= 2.0 + 1e-4) and np.all(a >= -2.0 - 1e-4))
        self.assertTrue(np.all(ad <= 2.0 + 1e-4) and np.all(ad >= -2.0 - 1e-4))

    def test_update_changes_policy_and_respects_kl(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        agent = _make_agent(self.act_space, hidden_sizes=(16,), max_kl=0.01)
        before = [p.detach().clone() for p in agent.policy.parameters()]
        stats = agent.update(_batch(agent))
        self.assertTrue(np.isfinite(stats["loss/policy"]))
        self.assertTrue(np.isfinite(stats["loss/value"]))
        self.assertTrue(np.isfinite(stats["stats/kl"]))
        if stats["stats/line_search_success"] == 1.0:
            self.assertLessEqual(stats["stats/kl"], 0.01 + 1e-6)
            self.assertTrue(any(
                not torch.allclose(b, a)
                for b, a in zip(before, agent.policy.parameters())
            ))

    def test_state_dict_round_trip(self) -> None:
        torch.manual_seed(1)
        agent = _make_agent(self.act_space)
        agent.update(_batch(agent))
        restored = _make_agent(self.act_space)
        restored.load_state_dict(agent.state_dict())
        for p, q in zip(agent.policy.parameters(), restored.policy.parameters()):
            self.assertTrue(torch.allclose(p, q))
        for p, q in zip(agent.value_fn.parameters(), restored.value_fn.parameters()):
            self.assertTrue(torch.allclose(p, q))


class TestTRPOAgentDiscrete(unittest.TestCase):
    def setUp(self) -> None:
        self.act_space = spaces.Discrete(3)

    def test_act_shapes_dtype_and_range(self) -> None:
        agent = _make_agent(self.act_space)
        self.assertTrue(agent.discrete)
        obs = np.zeros(5, dtype=np.float32)
        a, lp, v = agent.act(obs)
        ad, _, _ = agent.act(obs, deterministic=True)
        self.assertEqual(a.shape, (1,))
        self.assertEqual(a.dtype, np.int64)
        self.assertTrue(np.all(a >= 0) and np.all(a < 3))
        self.assertTrue(np.all(ad >= 0) and np.all(ad < 3))

    def test_update_changes_policy_and_respects_kl(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        agent = _make_agent(self.act_space, hidden_sizes=(16,), max_kl=0.01)
        before = [p.detach().clone() for p in agent.policy.parameters()]
        stats = agent.update(_batch(agent))
        self.assertTrue(np.isfinite(stats["loss/policy"]))
        self.assertTrue(np.isfinite(stats["stats/kl"]))
        if stats["stats/line_search_success"] == 1.0:
            self.assertLessEqual(stats["stats/kl"], 0.01 + 1e-6)
            self.assertTrue(any(
                not torch.allclose(b, a)
                for b, a in zip(before, agent.policy.parameters())
            ))

    def test_state_dict_round_trip(self) -> None:
        torch.manual_seed(1)
        agent = _make_agent(self.act_space)
        agent.update(_batch(agent))
        restored = _make_agent(self.act_space)
        restored.load_state_dict(agent.state_dict())
        for p, q in zip(agent.policy.parameters(), restored.policy.parameters()):
            self.assertTrue(torch.allclose(p, q))


class TestComputeGAE(unittest.TestCase):
    def test_gae_shapes_and_zero_reward(self) -> None:
        T, N = 4, 3
        rewards = np.zeros((T, N), dtype=np.float32)
        dones = np.zeros((T, N), dtype=np.float32)
        values = np.zeros((T, N), dtype=np.float32)
        next_value = np.zeros(N, dtype=np.float32)
        adv, ret = TRPOAgent.compute_gae(rewards, dones, values, next_value, 0.99, 0.95)
        self.assertEqual(adv.shape, (T, N))
        self.assertEqual(ret.shape, (T, N))
        # With zero rewards and zero values, advantages and returns are zero.
        self.assertTrue(np.allclose(adv, 0.0))
        self.assertTrue(np.allclose(ret, 0.0))

    def test_gae_single_step_reward(self) -> None:
        # One step, terminal, reward 1, zero values: advantage == reward.
        rewards = np.array([[1.0]], dtype=np.float32)
        dones = np.array([[1.0]], dtype=np.float32)
        values = np.array([[0.0]], dtype=np.float32)
        next_value = np.array([5.0], dtype=np.float32)  # ignored because done
        adv, ret = TRPOAgent.compute_gae(rewards, dones, values, next_value, 0.99, 0.95)
        self.assertAlmostEqual(float(adv[0, 0]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
