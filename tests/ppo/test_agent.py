"""PPO agent tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/ppo/test_agent.py
    >>> # Run with unittest
    >>> # python -m unittest tests.ppo.test_agent
"""

import unittest

import numpy as np
import torch

from PPO.ppo.agent import PPOAgent, Batch


class TestPPOAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.obs_dim = 4
        self.act_dim = 3
        self.agent = PPOAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=(8,),
            activation="tanh",
            lr=1e-3,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cpu",
        )

    def test_act_returns_numpy_arrays(self) -> None:
        obs = np.zeros((3, self.obs_dim), dtype=np.float32)

        actions, logprobs, values = self.agent.act(obs)

        self.assertEqual(actions.shape, (3,))
        self.assertEqual(logprobs.shape, (3,))
        self.assertEqual(values.shape, (3,))
        self.assertTrue(np.issubdtype(actions.dtype, np.integer))

    def test_compute_gae_terminal_returns_reward(self) -> None:
        rewards = np.array(
            [[1.0, 0.5], [0.0, 2.0], [-1.0, 0.0]], dtype=np.float32
        )
        dones = np.ones_like(rewards, dtype=np.float32)
        values = np.array(
            [[0.2, 0.4], [0.1, 1.0], [-0.5, 0.1]], dtype=np.float32
        )
        next_value = np.zeros(2, dtype=np.float32)

        advantages, returns = PPOAgent.compute_gae(
            rewards, dones, values, next_value, gamma=0.99, gae_lambda=0.95
        )

        expected_advantages = rewards - values
        np.testing.assert_allclose(advantages, expected_advantages, atol=1e-6)
        np.testing.assert_allclose(returns, rewards, atol=1e-6)

    def test_update_changes_parameters_and_returns_stats(self) -> None:
        torch.manual_seed(0)
        batch_size = 8
        obs = torch.randn((batch_size, self.obs_dim), dtype=torch.float32)
        actions = torch.randint(0, self.act_dim, (batch_size,), dtype=torch.int64)
        with torch.no_grad():
            logprob, _, values = self.agent.evaluate(obs, actions)
        advantages = torch.randn((batch_size,), dtype=torch.float32)
        returns = values + advantages

        batch = Batch(
            obs=obs,
            actions=actions,
            logprobs=logprob.detach(),
            returns=returns.detach(),
            advantages=advantages,
            values=values.detach(),
        )

        before = [p.detach().clone() for p in self.agent.model.parameters()]
        stats = self.agent.update(batch)

        changed = any(
            not torch.allclose(b, a) for b, a in zip(before, self.agent.model.parameters())
        )
        self.assertTrue(changed)
        self.assertIn("loss/total", stats)
        self.assertIn("loss/policy", stats)
        self.assertIn("loss/value", stats)
        self.assertIn("stats/entropy", stats)
        self.assertIn("stats/approx_kl", stats)
        for value in stats.values():
            self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
