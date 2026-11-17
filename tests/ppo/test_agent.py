"""Fast PPO action, optimization, and episode-boundary regressions."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from PPO.ppo.agent import Batch, PPOAgent


class TestPPOAgent(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.obs_dim = 4
        self.act_dim = 3
        self.agent = PPOAgent(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=(8,),
            activation="tanh",
            lr=1e-3,
            clip_coef=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cpu",
        )

    def test_act_returns_numpy_arrays_and_valid_actions(self) -> None:
        obs = np.zeros((3, self.obs_dim), dtype=np.float32)
        actions, logprobs, values = self.agent.act(obs)
        for result in (actions, logprobs, values):
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (3,))
            self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.issubdtype(actions.dtype, np.integer))
        self.assertTrue(((actions >= 0) & (actions < self.act_dim)).all())

    def test_deterministic_act_is_repeatable_argmax_without_sampling(self) -> None:
        with torch.no_grad():
            self.agent.model.pi.weight.zero_()
            self.agent.model.pi.bias.copy_(torch.tensor([-1.0, 2.0, 0.5]))
        obs = np.zeros((5, self.obs_dim), dtype=np.float32)
        with patch("torch.distributions.Categorical.sample") as sample:
            first = self.agent.act(obs, deterministic=True)
            second = self.agent.act(obs, deterministic=True)
        sample.assert_not_called()
        np.testing.assert_array_equal(first[0], np.ones(5, dtype=np.int64))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        obs_t = torch.as_tensor(obs)
        logprobs, _, values = self.agent.evaluate(obs_t, torch.as_tensor(first[0]))
        np.testing.assert_allclose(first[1], logprobs.detach().numpy(), atol=1e-6)
        np.testing.assert_allclose(first[2], values.detach().numpy(), atol=1e-6)

    def test_stochastic_act_uses_sampling_path(self) -> None:
        obs = np.zeros((3, self.obs_dim), dtype=np.float32)
        chosen = torch.tensor([0, 2, 1])
        with patch(
            "torch.distributions.Categorical.sample", return_value=chosen
        ) as sample:
            actions, _, _ = self.agent.act(obs, deterministic=False)
        sample.assert_called_once()
        np.testing.assert_array_equal(actions, chosen.numpy())

    def test_compute_gae_terminal_returns_reward(self) -> None:
        rewards = np.array([[1.0, 0.5], [0.0, 2.0], [-1.0, 0.0]], dtype=np.float32)
        dones = np.ones_like(rewards)
        values = np.array([[0.2, 0.4], [0.1, 1.0], [-0.5, 0.1]], dtype=np.float32)
        advantages, returns = PPOAgent.compute_gae(
            rewards, dones, values, np.array([9.0, 8.0]), 0.99, 0.95
        )
        np.testing.assert_allclose(advantages, rewards - values, atol=1e-6)
        np.testing.assert_allclose(returns, rewards, atol=1e-6)

    def test_compute_gae_uses_current_transition_done_not_next_done(self) -> None:
        rewards = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        dones = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
        values = np.array([[0.5], [1.0], [2.0]], dtype=np.float32)
        originals = [item.copy() for item in (rewards, dones, values)]
        advantages, returns = PPOAgent.compute_gae(
            rewards, dones, values, np.array([4.0]), 0.5, 1.0
        )
        # The terminal transition at t=1 cannot include the next episode's t=2.
        np.testing.assert_allclose(advantages, [[1.5], [1.0], [3.0]], atol=1e-6)
        np.testing.assert_allclose(returns, [[2.0], [2.0], [5.0]], atol=1e-6)
        for original, actual in zip(originals, (rewards, dones, values)):
            np.testing.assert_array_equal(original, actual)

    def test_compute_gae_timeout_bootstraps_but_stops_recurrence(self) -> None:
        rewards = np.array([[1.0], [100.0]], dtype=np.float32)
        dones = np.array([[1.0], [0.0]], dtype=np.float32)
        values = np.array([[2.0], [3.0]], dtype=np.float32)
        advantages, returns = PPOAgent.compute_gae(
            rewards,
            dones,
            values,
            np.array([999.0]),
            0.5,
            1.0,
            next_values=np.array([[10.0], [5.0]], dtype=np.float32),
            terminations=np.zeros_like(dones),
        )
        # Timeout gets final-observation V=10, not reset-state V=3 or next ep reward.
        np.testing.assert_allclose(advantages, [[4.0], [99.5]], atol=1e-6)
        np.testing.assert_allclose(returns, [[6.0], [102.5]], atol=1e-6)

    def test_compute_gae_final_step_handles_terminal_and_timeout_envs(self) -> None:
        rewards = np.array([[1.0, 2.0]], dtype=np.float32)
        values = np.array([[0.25, 0.5]], dtype=np.float32)
        advantages, returns = PPOAgent.compute_gae(
            rewards,
            np.ones_like(rewards),
            values,
            np.array([999.0, 999.0]),
            0.5,
            0.8,
            next_values=np.array([[100.0, 8.0]], dtype=np.float32),
            terminations=np.array([[1.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(advantages, [[0.75, 5.5]], atol=1e-6)
        np.testing.assert_allclose(returns, [[1.0, 6.0]], atol=1e-6)

    def test_compute_gae_nonterminal_rollout_bootstraps_multiple_envs(self) -> None:
        rewards = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        values = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
        advantages, returns = PPOAgent.compute_gae(
            rewards, np.zeros_like(rewards), values, np.array([2.5, 3.0]), 0.5, 0.8
        )
        np.testing.assert_allclose(advantages, [[2.35, 3.4], [2.75, 3.5]], atol=1e-6)
        np.testing.assert_allclose(returns, [[2.85, 4.4], [4.25, 5.5]], atol=1e-6)

    def _batch(self, batch_size: int) -> Batch:
        obs = torch.randn((batch_size, self.obs_dim))
        actions = torch.arange(batch_size) % self.act_dim
        with torch.no_grad():
            logprobs, _, values = self.agent.evaluate(obs, actions)
        advantages = torch.linspace(0.5, 1.5, batch_size)
        return Batch(
            obs,
            actions,
            logprobs.detach(),
            values.detach() + advantages,
            advantages,
            values.detach(),
        )

    def _assert_finite_update(self, batch_size: int) -> None:
        before = [
            parameter.detach().clone() for parameter in self.agent.model.parameters()
        ]
        stats = self.agent.update(self._batch(batch_size))
        self.assertTrue(
            any(
                not torch.allclose(left, right)
                for left, right in zip(before, self.agent.model.parameters())
            )
        )
        self.assertEqual(
            set(stats),
            {
                "loss/total",
                "loss/policy",
                "loss/value",
                "stats/entropy",
                "stats/approx_kl",
            },
        )
        for value in stats.values():
            self.assertTrue(np.isfinite(value))
        for parameter in self.agent.model.parameters():
            self.assertTrue(torch.isfinite(parameter).all().item())
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_update_changes_parameters_and_returns_finite_stats(self) -> None:
        self._assert_finite_update(8)

    def test_singleton_minibatch_update_has_finite_stats_and_gradients(self) -> None:
        self._assert_finite_update(1)


if __name__ == "__main__":
    unittest.main()
