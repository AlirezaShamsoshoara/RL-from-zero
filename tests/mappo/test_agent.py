"""MAPPO agent tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/mappo/test_agent.py
    >>> # Run with unittest
    >>> # python -m unittest tests.mappo.test_agent
"""

import unittest

import numpy as np
import torch

from MAPPO.mappo.agent import Batch, MAPPOAgent


class TestMAPPOAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.n_agents = 2
        self.obs_dim = 3
        self.act_dim = 2
        self.state_dim = self.obs_dim * self.n_agents
        self.agent = MAPPOAgent(
            n_agents=self.n_agents,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            state_dim=self.state_dim,
            actor_hidden_sizes=(8,),
            critic_hidden_sizes=(8,),
            activation="tanh",
            lr=1e-3,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cpu",
            share_policy=False,
            use_centralized_critic=True,
        )

    def test_act_returns_numpy_arrays(self) -> None:
        obs_list = [np.zeros(self.obs_dim, dtype=np.float32) for _ in range(self.n_agents)]
        state = np.zeros(self.state_dim, dtype=np.float32)

        actions, logprobs, values = self.agent.act(obs_list, state)

        self.assertEqual(actions.shape, (self.n_agents,))
        self.assertEqual(logprobs.shape, (self.n_agents,))
        self.assertEqual(values.shape, (self.n_agents,))
        self.assertTrue(np.issubdtype(actions.dtype, np.integer))

    def test_share_policy_uses_single_model(self) -> None:
        shared = MAPPOAgent(
            n_agents=3,
            obs_dim=2,
            act_dim=2,
            state_dim=6,
            actor_hidden_sizes=(4,),
            critic_hidden_sizes=(4,),
            activation="relu",
            lr=1e-3,
            clip_coef=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cpu",
            share_policy=True,
            use_centralized_critic=True,
        )

        self.assertIs(shared.models[0], shared.models[1])
        self.assertEqual(len(shared.optimizers), 1)

    def test_compute_gae_terminal_returns_reward(self) -> None:
        rewards = np.array([[1.0, 0.5], [0.0, 2.0], [-1.0, 0.0]], dtype=np.float32)
        dones = np.ones_like(rewards, dtype=np.float32)
        values = np.array([[0.2, 0.4], [0.1, 1.0], [-0.5, 0.1]], dtype=np.float32)
        next_values = np.zeros(2, dtype=np.float32)

        advantages, returns = MAPPOAgent.compute_gae(
            rewards, dones, values, next_values, gamma=0.99, gae_lambda=0.95
        )

        expected_advantages = rewards - values
        np.testing.assert_allclose(advantages, expected_advantages, atol=1e-6)
        np.testing.assert_allclose(returns, rewards, atol=1e-6)

    def test_compute_gae_does_not_bootstrap_across_episode_boundary(self) -> None:
        # t=0 is terminal, t=1 is next episode. Return at t=0 should not include
        # value/reward information from t=1.
        rewards = np.array([[1.0], [0.0]], dtype=np.float32)
        dones = np.array([[1.0], [0.0]], dtype=np.float32)
        values = np.array([[0.5], [0.2]], dtype=np.float32)
        next_values = np.array([0.3], dtype=np.float32)

        advantages, returns = MAPPOAgent.compute_gae(
            rewards, dones, values, next_values, gamma=0.99, gae_lambda=0.95
        )

        np.testing.assert_allclose(advantages[0, 0], rewards[0, 0] - values[0, 0], atol=1e-6)
        np.testing.assert_allclose(returns[0, 0], rewards[0, 0], atol=1e-6)

    def test_update_changes_parameters_and_returns_stats(self) -> None:
        torch.manual_seed(0)
        batch_size = 6
        batches = []

        for agent_id in range(self.n_agents):
            obs = torch.randn((batch_size, self.obs_dim), dtype=torch.float32)
            state = torch.randn((batch_size, self.state_dim), dtype=torch.float32)
            actions = torch.randint(0, self.act_dim, (batch_size,), dtype=torch.int64)
            with torch.no_grad():
                logprob, _ = self.agent.models[agent_id].evaluate_actions(obs, actions)
                values = self.agent.models[agent_id].get_value(state)
            advantages = torch.randn((batch_size,), dtype=torch.float32)
            returns = values + advantages

            batches.append(
                Batch(
                    obs=obs,
                    state=state,
                    actions=actions,
                    logprobs=logprob.detach(),
                    returns=returns.detach(),
                    advantages=advantages,
                    values=values.detach(),
                    alive_mask=torch.ones((batch_size,), dtype=torch.float32),
                )
            )

        before = [p.detach().clone() for p in self.agent.models[0].parameters()]
        stats, update_steps = self.agent.update(batches)

        changed = any(
            not torch.allclose(b, a)
            for b, a in zip(before, self.agent.models[0].parameters())
        )
        self.assertTrue(changed)
        self.assertGreater(update_steps, 0)
        self.assertIn("loss/total", stats)
        self.assertIn("loss/policy", stats)
        self.assertIn("loss/value", stats)
        self.assertIn("stats/entropy", stats)
        self.assertIn("stats/approx_kl", stats)
        for value in stats.values():
            self.assertTrue(np.isfinite(value))


if __name__ == "__main__":
    unittest.main()
