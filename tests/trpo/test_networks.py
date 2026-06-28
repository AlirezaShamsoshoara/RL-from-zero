"""TRPO network tests.

Examples:
    >>> # python -m pytest tests/trpo/test_networks.py
    >>> # python -m unittest tests.trpo.test_networks
"""

import unittest

import torch

from TRPO.trpo.networks import (
    CategoricalPolicy,
    GaussianPolicy,
    ValueNetwork,
    build_mlp,
)


class TestNetworks(unittest.TestCase):
    def test_build_mlp_invalid_activation(self) -> None:
        with self.assertRaises(ValueError):
            build_mlp(4, [8], activation="unknown")

    def test_gaussian_policy_shapes_bounds_and_stats(self) -> None:
        torch.manual_seed(0)
        low = torch.tensor([-2.0, -1.0], dtype=torch.float32)
        high = torch.tensor([2.0, 1.0], dtype=torch.float32)
        pol = GaussianPolicy(3, 2, (8,), "tanh", low, high)
        obs = torch.randn(5, 3)

        action, log_prob, entropy, mean_action = pol.sample(obs)
        self.assertEqual(action.shape, (5, 2))
        self.assertEqual(log_prob.shape, (5,))
        self.assertEqual(entropy.shape, (5,))
        self.assertTrue((action <= high + 1e-4).all().item())
        self.assertTrue((action >= low - 1e-4).all().item())

        greedy = pol.greedy(obs)
        self.assertEqual(greedy.shape, (5, 2))
        self.assertEqual(pol.log_prob(obs, action).shape, (5,))
        self.assertEqual(pol.entropy(obs).shape, (5,))

        old_mean, old_log_std = pol.detached_params(obs)
        with torch.no_grad():
            kl = pol.kl(obs, old_mean, old_log_std)
        self.assertEqual(kl.shape, (5,))
        # KL against the same params is ~0.
        self.assertLess(float(kl.mean()), 1e-5)
        self.assertFalse(pol.is_discrete)

    def test_categorical_policy_shapes_and_stats(self) -> None:
        torch.manual_seed(0)
        pol = CategoricalPolicy(4, 3, (8,), "tanh")
        obs = torch.randn(6, 4)

        logits = pol(obs)
        self.assertEqual(logits.shape, (6, 3))

        action, log_prob, entropy, greedy = pol.sample(obs)
        self.assertEqual(action.shape, (6,))
        self.assertEqual(action.dtype, torch.int64)
        self.assertEqual(log_prob.shape, (6,))
        self.assertEqual(entropy.shape, (6,))
        self.assertEqual(greedy.shape, (6,))
        self.assertTrue((action >= 0).all().item() and (action < 3).all().item())

        self.assertEqual(pol.log_prob(obs, action).shape, (6,))
        (old_logits,) = pol.detached_params(obs)
        with torch.no_grad():
            kl = pol.kl(obs, old_logits)
        self.assertEqual(kl.shape, (6,))
        self.assertLess(float(kl.mean()), 1e-6)
        self.assertTrue(pol.is_discrete)

    def test_value_network_shape(self) -> None:
        vf = ValueNetwork(3, (8,), "relu")
        self.assertEqual(vf(torch.randn(7, 3)).shape, (7,))


if __name__ == "__main__":
    unittest.main()
