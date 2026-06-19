"""IQL network tests.

Examples:
    >>> # python -m pytest tests/iql/test_networks.py
    >>> # python -m unittest tests.iql.test_networks
"""

import unittest

import torch

from IQL.iql.networks import (
    GaussianPolicy,
    QNetwork,
    ValueNetwork,
    build_mlp,
    get_activation,
)


class TestNetworks(unittest.TestCase):
    def test_build_mlp_and_activation_errors(self) -> None:
        with self.assertRaises(ValueError):
            build_mlp(4, [8], activation="unknown")
        with self.assertRaises(ValueError):
            get_activation("nope")
        self.assertIsNotNone(get_activation("relu"))

    def test_gaussian_policy_shapes_bounds_and_logprob(self) -> None:
        torch.manual_seed(0)
        low = torch.tensor([-2.0], dtype=torch.float32)
        high = torch.tensor([2.0], dtype=torch.float32)
        pol = GaussianPolicy(3, 1, (16,), "relu", low, high)
        obs = torch.randn(5, 3)

        action, log_prob, mean_action = pol.sample(obs)
        self.assertEqual(action.shape, (5, 1))
        self.assertEqual(log_prob.shape, (5, 1))
        self.assertTrue((action <= high + 1e-4).all().item())
        self.assertTrue((action >= low - 1e-4).all().item())

        det = pol.deterministic(obs)
        self.assertEqual(det.shape, (5, 1))
        self.assertTrue((det <= high + 1e-4).all().item())
        self.assertTrue((det >= low - 1e-4).all().item())

        lp = pol.log_prob(obs, action)
        self.assertEqual(lp.shape, (5, 1))
        self.assertTrue(torch.isfinite(lp).all().item())

    def test_q_and_value_network_shapes(self) -> None:
        q = QNetwork(3, 2, (8,), "tanh")
        self.assertEqual(q(torch.randn(6, 3), torch.randn(6, 2)).shape, (6, 1))
        v = ValueNetwork(3, (8,), "relu")
        self.assertEqual(v(torch.randn(6, 3)).shape, (6, 1))


if __name__ == "__main__":
    unittest.main()
