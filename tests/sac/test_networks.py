"""SAC network tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/sac/test_networks.py
    >>> # Run with unittest
    >>> # python -m unittest tests.sac.test_networks
"""

import math
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

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

    def test_activation_is_case_insensitive_and_instances_are_not_shared(self) -> None:
        self.assertIsInstance(get_activation("ReLU"), torch.nn.ReLU)
        self.assertIsNot(get_activation("relu"), get_activation("relu"))

    def test_mlp_consumes_hidden_iterator_and_has_independent_activations(self) -> None:
        net = build_mlp(3, iter([7, 5]), activation="tanh")
        self.assertEqual(net(torch.zeros(4, 3)).shape, (4, 5))
        self.assertEqual(
            [
                (m.in_features, m.out_features)
                for m in net
                if isinstance(m, torch.nn.Linear)
            ],
            [(3, 7), (7, 5)],
        )
        self.assertIsNot(net[1], net[3])

    def test_empty_hidden_sizes_preserve_linear_policy_and_critic(self) -> None:
        policy = self._policy(hidden_sizes=())
        qnet = QNetwork(3, 2, (), "relu")
        self.assertEqual(policy.mean_layer.in_features, 3)
        self.assertEqual(qnet.head.in_features, 5)
        self.assertEqual(policy.deterministic(torch.zeros(4, 3)).shape, (4, 2))
        self.assertEqual(qnet(torch.zeros(4, 3), torch.zeros(4, 2)).shape, (4, 1))

    def test_policy_and_critic_accept_single_pass_hidden_iterators(self) -> None:
        policy = self._policy(hidden_sizes=iter([7, 5]))
        critic = QNetwork(3, 2, iter([7, 5]), "elu")
        self.assertEqual(policy(torch.zeros(4, 3))[0].shape, (4, 2))
        self.assertEqual(critic(torch.zeros(4, 3), torch.zeros(4, 2)).shape, (4, 1))

    def test_log_standard_deviation_is_clamped_to_configured_bounds(self) -> None:
        policy = self._policy(log_std_bounds=(-3.0, 1.0))
        with torch.no_grad():
            policy.log_std_layer.weight.zero_()
            policy.log_std_layer.bias.copy_(torch.tensor([-100.0, 100.0]))
        _, log_std = policy(torch.zeros(3, 3))
        torch.testing.assert_close(log_std, torch.tensor([[-3.0, 1.0]]).expand(3, 2))

    def test_physical_action_log_density_includes_affine_jacobian(self) -> None:
        policy = self._policy(
            action_low=torch.tensor([-3.0, 1.0]), action_high=torch.tensor([1.0, 7.0])
        )
        with torch.no_grad():
            for param in policy.parameters():
                param.zero_()
        z = torch.tensor([[0.25, -0.5], [0.75, 0.125]])
        with patch("torch.distributions.Normal.rsample", return_value=z):
            actions, log_prob, _ = policy.sample(torch.zeros(2, 3))
        scales = torch.tensor([2.0, 3.0])
        expected = (
            -0.5 * (z.square() + math.log(2.0 * math.pi))
            - scales.log()
            - torch.log1p(-torch.tanh(z).square())
        ).sum(-1, keepdim=True)
        torch.testing.assert_close(log_prob, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            actions, torch.tanh(z) * scales + torch.tensor([-1.0, 4.0])
        )

    def test_tanh_saturation_has_exact_finite_density_and_gradients(self) -> None:
        policy = self._policy()
        with torch.no_grad():
            for param in policy.parameters():
                param.zero_()
            policy.mean_layer.bias.copy_(torch.tensor([40.0, -40.0]))
        obs = torch.zeros(2, 3)
        mean, log_std = policy(obs)
        with patch("torch.distributions.Normal.rsample", return_value=mean):
            actions, log_prob, _ = policy.sample(obs)
        jacobian = 2.0 * (math.log(2.0) - mean - F.softplus(-2.0 * mean))
        expected = (
            -log_std
            - 0.5 * math.log(2.0 * math.pi)
            - policy.action_scale.log()
            - jacobian
        ).sum(-1, keepdim=True)
        torch.testing.assert_close(log_prob, expected)
        self.assertTrue(torch.isfinite(actions).all())
        log_prob.sum().backward()
        self.assertTrue(torch.isfinite(policy.mean_layer.bias.grad).all())
        self.assertGreater(policy.mean_layer.bias.grad.abs().sum().item(), 0.0)

    def test_action_bounds_must_be_finite_ordered_vectors(self) -> None:
        invalid = [
            (torch.tensor([-1.0, -1.0]), torch.tensor([-1.0, 1.0])),
            (torch.tensor([2.0, -1.0]), torch.tensor([1.0, 1.0])),
            (torch.tensor([-float("inf"), -1.0]), torch.ones(2)),
            (-torch.ones(2), torch.tensor([float("nan"), 1.0])),
            (-torch.ones(1), torch.ones(2)),
            (-torch.ones(1, 2), torch.ones(1, 2)),
        ]
        for low, high in invalid:
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                self._policy(action_low=low, action_high=high)

    def test_log_standard_deviation_bounds_must_be_ordered_and_finite(self) -> None:
        for bounds in [
            (1.0, -1.0),
            (1.0, 1.0),
            (-float("inf"), 2.0),
            (-2.0, float("nan")),
        ]:
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                self._policy(log_std_bounds=bounds)

    def test_network_dimensions_and_hidden_widths_are_positive_integers(self) -> None:
        for width in [0, -1, True, 2.5]:
            with self.subTest(width=width), self.assertRaises(ValueError):
                build_mlp(3, [width], "relu")
        for dim in [0, -1, True, 2.5]:
            with self.subTest(dim=dim), self.assertRaises(ValueError):
                QNetwork(dim, 2, (8,), "relu")

    @staticmethod
    def _policy(**kwargs):
        defaults = dict(
            obs_dim=3,
            act_dim=2,
            hidden_sizes=(8,),
            activation="relu",
            action_low=-torch.ones(2),
            action_high=torch.ones(2),
        )
        defaults.update(kwargs)
        return GaussianPolicy(**defaults)


if __name__ == "__main__":
    unittest.main()
