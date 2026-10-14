"""MAPPO agent tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/mappo/test_agent.py
    >>> # Run with unittest
    >>> # python -m unittest tests.mappo.test_agent
"""

import unittest
from unittest.mock import patch

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
            action_type="discrete",
        )

    def test_act_returns_numpy_arrays(self) -> None:
        obs_list = [
            np.zeros(self.obs_dim, dtype=np.float32) for _ in range(self.n_agents)
        ]
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

        np.testing.assert_allclose(
            advantages[0, 0], rewards[0, 0] - values[0, 0], atol=1e-6
        )
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


def small_agent(shared=True, centralized=True, action_type="continuous"):
    return MAPPOAgent(
        2,
        2,
        2,
        4,
        (4,),
        (4,),
        "tanh",
        0.001,
        0.2,
        0.01,
        0.5,
        0.5,
        "cpu",
        share_policy=shared,
        use_centralized_critic=centralized,
        action_type=action_type,
    )


def agent_batches(agent, size=2):
    batches = []
    for index in range(agent.n_agents):
        obs, state = torch.randn((size, 2)), torch.randn((size, 4))
        with torch.no_grad():
            actions, logprobs, _ = agent.models[index].act(obs, deterministic=True)
            values = agent.models[index].get_value(
                state if agent.use_centralized_critic else obs
            )
        advantages = torch.arange(size, dtype=torch.float32) + 0.5 + index
        batches.append(
            Batch(
                obs,
                state,
                actions.detach(),
                logprobs.detach(),
                values.detach() + advantages,
                advantages,
                values.detach(),
                torch.ones(size),
            )
        )
    return batches


class TestAgentContracts(unittest.TestCase):
    def test_continuous_actions_and_readonly_values_use_no_rng(self):
        agent = small_agent()
        observations = [np.zeros(2, dtype=np.float32)] * 2
        state = np.zeros(4, dtype=np.float32)
        with (
            patch(
                "torch.distributions.Normal.rsample",
                side_effect=AssertionError("RNG used"),
            ),
            patch(
                "torch.distributions.Normal.sample",
                side_effect=AssertionError("RNG used"),
            ),
            patch(
                "torch.distributions.Categorical.sample",
                side_effect=AssertionError("RNG used"),
            ),
        ):
            first = agent.act(observations, state, deterministic=True)
            second = agent.act(observations, state, deterministic=True)
            values = agent.values(observations, state)
        self.assertEqual(first[0].shape, (2, 2))
        self.assertEqual(first[1].shape, (2,))
        self.assertEqual(values.shape, (2,))
        self.assertTrue((np.abs(first[0]) <= 1.0).all())
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_allclose(values, first[2])

    def test_centralized_values_require_state_and_decentralized_values_do_not(self):
        observations = [np.zeros(2, dtype=np.float32)] * 2
        with self.assertRaises(ValueError):
            small_agent(centralized=True).values(observations)
        values = small_agent(centralized=False).values(observations)
        self.assertEqual(values.shape, (2,))
        self.assertTrue(np.isfinite(values).all())

    def test_shared_policy_concatenates_alive_rows_for_one_joint_optimizer_step(self):
        torch.manual_seed(3)
        agent = small_agent(shared=True)
        batches = agent_batches(agent)
        batches[1].alive_mask[1] = 0
        before = [
            parameter.detach().clone() for parameter in agent.models[0].parameters()
        ]
        with (
            patch.object(
                agent.optimizers[0], "step", wraps=agent.optimizers[0].step
            ) as step,
            patch.object(
                agent.models[0],
                "evaluate_actions",
                wraps=agent.models[0].evaluate_actions,
            ) as evaluate,
        ):
            stats, count = agent.update(batches)
        self.assertEqual(count, 1)
        step.assert_called_once()
        self.assertEqual(evaluate.call_args.args[0].shape[0], 3)
        self.assertTrue(
            any(
                not torch.equal(one, two)
                for one, two in zip(before, agent.models[0].parameters())
            )
        )
        for key in (
            "loss/policy",
            "loss/value",
            "loss/total",
            "stats/entropy",
            "stats/approx_kl",
        ):
            self.assertTrue(np.isfinite(stats[key]))

    def test_separate_policies_step_each_optimizer_and_singletons_remain_finite(self):
        agent = small_agent(shared=False)
        batches = agent_batches(agent, size=1)
        with (
            patch.object(
                agent.optimizers[0], "step", wraps=agent.optimizers[0].step
            ) as first,
            patch.object(
                agent.optimizers[1], "step", wraps=agent.optimizers[1].step
            ) as second,
        ):
            stats, count = agent.update(batches)
        self.assertEqual(count, 2)
        first.assert_called_once()
        second.assert_called_once()
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))

    def test_all_dead_batches_skip_invalid_rows_and_do_not_mutate_parameters(self):
        for shared in (True, False):
            with self.subTest(shared=shared):
                agent = small_agent(shared=shared)
                batches = agent_batches(agent, size=1)
                for batch in batches:
                    batch.alive_mask.zero_()
                    batch.obs.fill_(float("nan"))
                    batch.state.fill_(float("nan"))
                    batch.actions.fill_(float("nan"))
                    batch.returns.fill_(float("nan"))
                before = [
                    [parameter.detach().clone() for parameter in model.parameters()]
                    for model in agent.models
                ]
                stats, count = agent.update(batches)
                self.assertEqual(count, 0)
                self.assertTrue(all(value == 0.0 for value in stats.values()))
                for old, model in zip(before, agent.models):
                    for one, two in zip(old, model.parameters()):
                        torch.testing.assert_close(one, two, rtol=0, atol=0)
                self.assertTrue(
                    all(not optimizer.state for optimizer in agent.optimizers)
                )

    def test_dead_invalid_row_is_excluded_from_joint_loss(self):
        agent = small_agent()
        batches = agent_batches(agent)
        batches[0].alive_mask[1] = 0
        for value in (batches[0].obs, batches[0].state, batches[0].actions):
            value[1] = float("nan")
        batches[0].returns[1] = float("nan")
        stats, count = agent.update(batches)
        self.assertEqual(count, 1)
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))

    def test_batch_count_and_live_shapes_are_validated(self):
        agent = small_agent()
        with self.assertRaises(ValueError):
            agent.update(agent_batches(agent)[:1])
        batches = agent_batches(agent)
        batches[0].obs = torch.zeros((2, 3))
        with self.assertRaises((ValueError, RuntimeError)):
            agent.update(batches)

    def test_timeout_gae_bootstraps_final_values_but_stops_recursion(self):
        rewards = np.array([[1.0, 2.0], [100.0, 0.0]], dtype=np.float32)
        dones = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        values = np.array([[2.0, 0.5], [3.0, 0.0]], dtype=np.float32)
        terminations = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        advantages, returns = MAPPOAgent.compute_gae(
            rewards,
            dones,
            values,
            np.array([999.0, 999.0]),
            0.5,
            1.0,
            next_value_sequence=np.array([[10.0, 99.0], [5.0, 99.0]], dtype=np.float32),
            terminations=terminations,
        )
        np.testing.assert_allclose(advantages, [[4.0, 1.5], [99.5, 0.0]], atol=1e-6)
        np.testing.assert_allclose(returns, [[6.0, 2.0], [102.5, 0.0]], atol=1e-6)

    def test_nonterminal_final_step_bootstrap_for_multiple_agents(self):
        rewards = np.array([[1.0, 2.0]], dtype=np.float32)
        values = np.array([[0.5, 1.0]], dtype=np.float32)
        advantages, returns = MAPPOAgent.compute_gae(
            rewards, np.zeros_like(rewards), values, np.array([3.0, 4.0]), 0.5, 0.9
        )
        np.testing.assert_allclose(advantages, [[2.0, 3.0]], atol=1e-6)
        np.testing.assert_allclose(returns, [[2.5, 4.0]], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
