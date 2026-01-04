"""SAC network tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/sac/test_networks.py
    >>> # Run with unittest
    >>> # python -m unittest tests.sac.test_networks
"""

import unittest

import torch

from SAC.sac.networks import GaussianPolicy, QNetwork, build_mlp, get_activation


class TestNetworks(unittest.TestCase):
    def test_get_activation_supported(self) -> None:
        self.assertIsInstance(get_activation("relu"), torch.nn.ReLU)
        self.assertIsInstance(get_activation("tanh"), torch.nn.Tanh)
        self.assertIsInstance(get_activation("leaky_relu"), torch.nn.LeakyReLU)
        self.assertIsInstance(get_activation("elu"), torch.nn.ELU)

    def test_build_mlp_invalid_activation(self) -> None:
        with self.assertRaises(ValueError):
            build_mlp(4, [8], activation="unknown")

    def test_gaussian_policy_sample_shapes_and_bounds(self) -> None:
        torch.manual_seed(0)
        low = torch.tensor([-2.0, -1.0], dtype=torch.float32)
        high = torch.tensor([2.0, 1.0], dtype=torch.float32)
        policy = GaussianPolicy(
            obs_dim=3,
            act_dim=2,
            hidden_sizes=(8,),
            activation="relu",
            action_low=low,
            action_high=high,
        )
        obs = torch.zeros((4, 3), dtype=torch.float32)

        action, log_prob, mean_action = policy.sample(obs)
        deterministic = policy.deterministic(obs)

        self.assertEqual(action.shape, (4, 2))
        self.assertEqual(log_prob.shape, (4, 1))
        self.assertEqual(mean_action.shape, (4, 2))
        self.assertEqual(deterministic.shape, (4, 2))
        self.assertTrue(torch.isfinite(log_prob).all().item())
        self.assertTrue((action <= high + 1e-5).all().item())
        self.assertTrue((action >= low - 1e-5).all().item())
        self.assertTrue((deterministic <= high + 1e-5).all().item())
        self.assertTrue((deterministic >= low - 1e-5).all().item())

    def test_qnetwork_output_shape(self) -> None:
        qnet = QNetwork(obs_dim=3, act_dim=2, hidden_sizes=(8,), activation="tanh")
        obs = torch.randn((5, 3), dtype=torch.float32)
        act = torch.randn((5, 2), dtype=torch.float32)

        q_values = qnet(obs, act)

        self.assertEqual(q_values.shape, (5, 1))


if __name__ == "__main__":
    unittest.main()
