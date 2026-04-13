"""A3C network tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/a3c/test_networks.py
    >>> # Run with unittest
    >>> # python -m unittest tests.a3c.test_networks
"""

import unittest

import torch

from A3C.a3c.networks import ActorCritic, get_activation


class TestNetworks(unittest.TestCase):
    def test_get_activation_returns_correct_types(self) -> None:
        self.assertIsInstance(get_activation("relu"), torch.nn.ReLU)
        self.assertIsInstance(get_activation("elu"), torch.nn.ELU)
        self.assertIsInstance(get_activation("leaky_relu"), torch.nn.LeakyReLU)
        self.assertIsInstance(get_activation("tanh"), torch.nn.Tanh)
        # unknown defaults to tanh
        self.assertIsInstance(get_activation("unknown"), torch.nn.Tanh)

    def test_forward_output_shapes(self) -> None:
        torch.manual_seed(0)
        model = ActorCritic(obs_dim=6, act_dim=3, hidden_sizes=(8,), activation="relu")
        obs = torch.randn((4, 6), dtype=torch.float32)

        logits, values = model.forward(obs)

        self.assertEqual(logits.shape, (4, 3))
        self.assertEqual(values.shape, (4,))

    def test_forward_single_obs_unsqueezes(self) -> None:
        torch.manual_seed(0)
        model = ActorCritic(obs_dim=6, act_dim=3, hidden_sizes=(8,), activation="relu")
        obs = torch.randn(6, dtype=torch.float32)

        logits, values = model.forward(obs)

        self.assertEqual(logits.shape, (1, 3))
        self.assertEqual(values.shape, (1,))

    def test_act_returns_correct_shapes(self) -> None:
        torch.manual_seed(0)
        model = ActorCritic(obs_dim=6, act_dim=3, hidden_sizes=(8,), activation="relu")
        obs = torch.randn(6, dtype=torch.float32)

        action, log_prob, entropy, value = model.act(obs)

        self.assertEqual(action.shape, (1,))
        self.assertEqual(log_prob.shape, (1,))
        self.assertEqual(entropy.shape, (1,))
        self.assertEqual(value.shape, (1,))
        # action should be a valid discrete action
        self.assertTrue(0 <= int(action.item()) < 3)

    def test_evaluate_actions_consistency(self) -> None:
        torch.manual_seed(0)
        model = ActorCritic(obs_dim=6, act_dim=3, hidden_sizes=(8,), activation="relu")
        obs = torch.randn((4, 6), dtype=torch.float32)
        actions = torch.tensor([0, 1, 2, 0], dtype=torch.long)

        log_probs, entropies, values = model.evaluate_actions(obs, actions)

        self.assertEqual(log_probs.shape, (4,))
        self.assertEqual(entropies.shape, (4,))
        self.assertEqual(values.shape, (4,))
        # log_probs should be negative
        self.assertTrue((log_probs <= 0).all().item())
        # entropies should be non-negative
        self.assertTrue((entropies >= 0).all().item())


if __name__ == "__main__":
    unittest.main()
