"""Actor-critic shape, distribution, and deterministic action tests."""

import unittest
from unittest.mock import patch

import torch

from PPO.ppo.networks import ActorCritic, get_activation


class TestNetworks(unittest.TestCase):
    def test_get_activation_defaults_to_tanh(self) -> None:
        self.assertIsInstance(get_activation("relu"), torch.nn.ReLU)
        self.assertIsInstance(get_activation(None), torch.nn.Tanh)
        self.assertIsInstance(get_activation("tanh"), torch.nn.Tanh)

    def test_actorcritic_forward_shapes(self) -> None:
        model = ActorCritic(obs_dim=4, act_dim=3, hidden_sizes=(8,))
        logits, values = model(torch.zeros((5, 4)))
        self.assertEqual(logits.shape, (5, 3))
        self.assertEqual(values.shape, (5,))
        self.assertTrue(torch.isfinite(logits).all().item())
        self.assertTrue(torch.isfinite(values).all().item())

    def test_actorcritic_sampled_actions_match_evaluate(self) -> None:
        torch.manual_seed(0)
        model = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=(8,))
        obs = torch.randn((6, 4))
        actions, logprob, entropy, values = model.act(obs)
        eval_logprob, eval_entropy, eval_values = model.evaluate_actions(obs, actions)
        self.assertTrue(((actions >= 0) & (actions < 2)).all().item())
        torch.testing.assert_close(logprob, eval_logprob)
        torch.testing.assert_close(values, eval_values)
        torch.testing.assert_close(entropy, eval_entropy)
        self.assertGreater(entropy.item(), 0.0)

    def test_deterministic_actions_use_argmax_and_do_not_sample(self) -> None:
        model = ActorCritic(obs_dim=4, act_dim=3, hidden_sizes=())
        with torch.no_grad():
            model.pi.weight.zero_()
            model.pi.bias.copy_(torch.tensor([-1.0, 2.0, 2.0]))
        obs = torch.randn((4, 4))
        with patch("torch.distributions.Categorical.sample") as sample:
            first = model.act(obs, deterministic=True)
            second = model.act(obs, deterministic=True)
        sample.assert_not_called()
        # Tied logits have a reproducible first-index argmax, not random tie-breaking.
        torch.testing.assert_close(first[0], torch.ones(4, dtype=torch.long))
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        logprob, entropy, value = model.evaluate_actions(obs, first[0])
        for actual, expected in zip(first[1:], (logprob, entropy, value)):
            torch.testing.assert_close(actual, expected)

    def test_explicit_stochastic_mode_samples_categorical(self) -> None:
        model = ActorCritic(obs_dim=4, act_dim=3, hidden_sizes=(8,))
        chosen = torch.tensor([2, 0])
        with patch(
            "torch.distributions.Categorical.sample", return_value=chosen
        ) as sample:
            actions, logprob, entropy, value = model.act(
                torch.zeros((2, 4)), deterministic=False
            )
        sample.assert_called_once()
        torch.testing.assert_close(actions, chosen)
        self.assertEqual(logprob.shape, (2,))
        self.assertEqual(value.shape, (2,))
        self.assertTrue(torch.isfinite(entropy).item())


if __name__ == "__main__":
    unittest.main()
