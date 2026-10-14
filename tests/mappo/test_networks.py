"""MAPPO network tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/mappo/test_networks.py
    >>> # Run with unittest
    >>> # python -m unittest tests.mappo.test_networks
"""

import unittest
from unittest.mock import patch

import torch

from MAPPO.mappo.networks import (
    ActorCritic,
    Actor,
    GaussianActor,
    CentralizedCritic,
    DecentralizedCritic,
    get_activation,
)


class TestNetworks(unittest.TestCase):
    def test_get_activation_defaults_to_tanh(self) -> None:
        self.assertIsInstance(get_activation("relu"), torch.nn.ReLU)
        self.assertIsInstance(get_activation(None), torch.nn.Tanh)
        self.assertIsInstance(get_activation("tanh"), torch.nn.Tanh)

    def test_actor_action_evaluate_match(self) -> None:
        torch.manual_seed(0)
        model = ActorCritic(
            obs_dim=4,
            act_dim=3,
            state_dim=8,
            actor_hidden_sizes=(8,),
            critic_hidden_sizes=(8,),
            activation="tanh",
            use_centralized_critic=True,
            action_type="discrete",
        )
        obs = torch.randn((6, 4), dtype=torch.float32)

        actions, logprob, entropy = model.act(obs)
        eval_logprob, eval_entropy = model.evaluate_actions(obs, actions)

        self.assertTrue(torch.allclose(logprob, eval_logprob, atol=1e-6))
        self.assertAlmostEqual(
            float(entropy.mean().item()), float(eval_entropy.mean().item()), places=6
        )
        self.assertEqual(actions.shape, (6,))

    def test_critic_value_shapes(self) -> None:
        obs = torch.zeros((5, 4), dtype=torch.float32)
        state = torch.zeros((5, 8), dtype=torch.float32)

        centralized = ActorCritic(
            obs_dim=4,
            act_dim=2,
            state_dim=8,
            actor_hidden_sizes=(8,),
            critic_hidden_sizes=(8,),
            activation="relu",
            use_centralized_critic=True,
            action_type="discrete",
        )
        decentralized = ActorCritic(
            obs_dim=4,
            act_dim=2,
            state_dim=8,
            actor_hidden_sizes=(8,),
            critic_hidden_sizes=(8,),
            activation="relu",
            use_centralized_critic=False,
        )

        value_c = centralized.get_value(state)
        value_d = decentralized.get_value(obs)

        self.assertIsInstance(centralized.critic, CentralizedCritic)
        self.assertIsInstance(decentralized.critic, DecentralizedCritic)
        self.assertEqual(value_c.shape, (5,))
        self.assertEqual(value_d.shape, (5,))


class TestContinuousNetworks(unittest.TestCase):
    def actor(self, low=None, high=None):
        result = GaussianActor(2, 2, hidden_sizes=(), action_low=low, action_high=high)
        with torch.no_grad():
            result.mu.weight.zero_()
            result.mu.bias.zero_()
        return result

    def test_explicit_continuous_actorcritic_and_invalid_action_type(self):
        model = ActorCritic(
            2,
            2,
            4,
            actor_hidden_sizes=(),
            critic_hidden_sizes=(),
            action_type="continuous",
        )
        self.assertIsInstance(model.actor, GaussianActor)
        self.assertIsInstance(ActorCritic(2, 2, 4, action_type="discrete").actor, Actor)
        with self.assertRaises(ValueError):
            ActorCritic(2, 2, 4, action_type="unknown")

    def test_asymmetric_scaled_logprob_contains_action_scale_jacobian(self):
        unit = self.actor()
        scaled = self.actor(low=[-2.0, 1.0], high=[2.0, 9.0])
        obs = torch.zeros((3, 2))
        action, logprob, entropy = scaled.get_action(obs, deterministic=True)
        _, unit_logprob, _ = unit.get_action(obs, deterministic=True)
        torch.testing.assert_close(action, torch.tensor([[0.0, 5.0]] * 3))
        torch.testing.assert_close(
            logprob, unit_logprob - torch.log(torch.tensor(8.0)), atol=1e-5, rtol=1e-5
        )
        evaluated, evaluated_entropy = scaled.evaluate_actions(obs, action)
        torch.testing.assert_close(logprob, evaluated, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isfinite(entropy).all().item())
        self.assertTrue(torch.isfinite(evaluated_entropy).all().item())

    def test_deterministic_squashed_mean_uses_no_rng_and_respects_bounds(self):
        actor = self.actor(low=[-2.0, 1.0], high=[2.0, 9.0])
        with torch.no_grad():
            actor.mu.bias.copy_(torch.tensor([0.5, -0.3]))
        obs = torch.zeros((2, 2))
        with (
            patch(
                "torch.distributions.Normal.rsample",
                side_effect=AssertionError("RNG used"),
            ),
            patch(
                "torch.distributions.Normal.sample",
                side_effect=AssertionError("RNG used"),
            ),
        ):
            first = actor.get_action(obs, deterministic=True)
            second = actor.get_action(obs, deterministic=True)
        expected = torch.tanh(torch.tensor([0.5, -0.3])) * torch.tensor(
            [2.0, 4.0]
        ) + torch.tensor([0.0, 5.0])
        torch.testing.assert_close(first[0], expected.repeat(2, 1))
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertTrue((first[0] >= torch.tensor([-2.0, 1.0])).all().item())
        self.assertTrue((first[0] <= torch.tensor([2.0, 9.0])).all().item())

    def test_saturated_actions_have_finite_consistent_get_and_evaluate_logprobs(self):
        actor = self.actor()
        with torch.no_grad():
            actor.mu.bias.copy_(torch.tensor([12.0, -12.0]))
        obs = torch.zeros((2, 2))
        actions, logprobs, _ = actor.get_action(obs, deterministic=True)
        evaluated, _ = actor.evaluate_actions(obs, actions)
        self.assertTrue(torch.isfinite(logprobs).all().item())
        self.assertTrue(torch.isfinite(evaluated).all().item())
        torch.testing.assert_close(logprobs, evaluated, atol=1e-4, rtol=1e-4)

    def test_stochastic_gaussian_samples_bounded_finite_actions(self):
        actor = self.actor()
        with patch(
            "torch.distributions.Normal.sample",
            return_value=torch.tensor([[1.0, -1.0]]),
        ) as sample:
            actions, logprobs, _ = actor.get_action(torch.zeros((1, 2)))
        sample.assert_called_once()
        self.assertTrue((actions.abs() <= 1.0).all().item())
        self.assertTrue(torch.isfinite(logprobs).all().item())

    def test_invalid_continuous_bounds_are_rejected(self):
        for low, high in (
            ([0.0, -1.0], [0.0, 1.0]),
            ([-float("inf"), -1.0], [1.0, 1.0]),
        ):
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                self.actor(low, high)


if __name__ == "__main__":
    unittest.main()
