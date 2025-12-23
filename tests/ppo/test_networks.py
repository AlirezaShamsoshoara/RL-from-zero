"""PPO network tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/ppo/test_networks.py
    >>> # Run with unittest
    >>> # python -m unittest tests.ppo.test_networks
"""

import unittest

import torch

from PPO.ppo.networks import ActorCritic, get_activation


class TestNetworks(unittest.TestCase):
    def test_get_activation_defaults_to_tanh(self) -> None:
        self.assertIsInstance(get_activation("relu"), torch.nn.ReLU)
        self.assertIsInstance(get_activation(None), torch.nn.Tanh)
        self.assertIsInstance(get_activation("tanh"), torch.nn.Tanh)

    def test_actorcritic_forward_shapes(self) -> None:
        model = ActorCritic(obs_dim=4, act_dim=3, hidden_sizes=(8,))
        obs = torch.zeros((5, 4), dtype=torch.float32)

        logits, values = model(obs)

        self.assertEqual(logits.shape, (5, 3))
        self.assertEqual(values.shape, (5,))

    def test_actorcritic_act_matches_evaluate(self) -> None:
        torch.manual_seed(0)
        model = ActorCritic(obs_dim=4, act_dim=2, hidden_sizes=(8,))
        obs = torch.randn((6, 4), dtype=torch.float32)

        actions, logprob, entropy, values = model.act(obs)
        eval_logprob, eval_entropy, eval_values = model.evaluate_actions(obs, actions)

        self.assertTrue(torch.allclose(logprob, eval_logprob, atol=1e-6))
        self.assertEqual(values.shape, eval_values.shape)
        self.assertGreater(entropy.item(), 0.0)
        self.assertAlmostEqual(entropy.item(), eval_entropy.item(), places=6)


if __name__ == "__main__":
    unittest.main()
