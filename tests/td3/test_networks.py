"""TD3 network tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/td3/test_networks.py
    >>> # Run with unittest
    >>> # python -m unittest tests.td3.test_networks
"""

import unittest

import torch

from TD3.td3.networks import Actor, Critic, build_mlp


class TestNetworks(unittest.TestCase):
    def test_build_mlp_invalid_activation(self) -> None:
        with self.assertRaises(ValueError):
            build_mlp(4, [8], activation="unknown")

    def test_actor_output_shape_and_bounds(self) -> None:
        torch.manual_seed(0)
        low = torch.tensor([-2.0, -1.0], dtype=torch.float32)
        high = torch.tensor([2.0, 1.0], dtype=torch.float32)
        actor = Actor(
            obs_dim=3,
            act_dim=2,
            hidden_sizes=(8,),
            activation="relu",
            action_low=low,
            action_high=high,
        )
        obs = torch.zeros((4, 3), dtype=torch.float32)

        actions = actor(obs)

        self.assertEqual(actions.shape, (4, 2))
        self.assertTrue((actions <= high + 1e-5).all().item())
        self.assertTrue((actions >= low - 1e-5).all().item())

    def test_critic_output_shape(self) -> None:
        critic = Critic(obs_dim=3, act_dim=2, hidden_sizes=(8,), activation="tanh")
        obs = torch.randn((5, 3), dtype=torch.float32)
        act = torch.randn((5, 2), dtype=torch.float32)

        q_values = critic(obs, act)

        self.assertEqual(q_values.shape, (5, 1))


class TestNetworkContracts(unittest.TestCase):
    def test_empty_hidden_sizes_are_linear_models(self):
        actor = Actor(
            3, 2, [], "relu", torch.tensor([-2.0, 1.0]), torch.tensor([2.0, 3.0])
        )
        critic = Critic(3, 2, [], "relu")
        self.assertEqual(actor(torch.zeros(5, 3)).shape, (5, 2))
        self.assertEqual(critic(torch.zeros(5, 3), torch.zeros(5, 2)).shape, (5, 1))
        self.assertEqual(len(build_mlp(3, [], "relu")), 0)

    def test_asymmetric_and_tiny_bounds_have_exact_midpoint_and_scale(self):
        low = torch.tensor([-1e-8, 2.0])
        high = torch.tensor([1e-8, 5.0])
        actor = Actor(3, 2, (), "relu", low, high)
        with torch.no_grad():
            actor.head.weight.zero_()
            actor.head.bias.zero_()
        torch.testing.assert_close(
            actor(torch.zeros(4, 3)), ((low + high) / 2).expand(4, 2)
        )
        with torch.no_grad():
            actor.head.bias.fill_(100)
        actions = actor(torch.zeros(4, 3))
        self.assertTrue(torch.all(actions <= high).item())
        self.assertTrue(torch.all(actions >= low).item())
        torch.testing.assert_close(actions, high.expand(4, 2), rtol=0, atol=0)

    def test_invalid_widths_dimensions_and_bounds_are_rejected(self):
        for input_dim, hidden in [
            (0, [8]),
            (True, [8]),
            (3, [0]),
            (3, [1.5]),
            (3, [True]),
        ]:
            with (
                self.subTest(input_dim=input_dim, hidden=hidden),
                self.assertRaises((ValueError, TypeError)),
            ):
                build_mlp(input_dim, hidden, "relu")
        for low, high in [
            (torch.zeros(2), torch.zeros(2)),
            (torch.ones(2), torch.zeros(2)),
            (torch.zeros(1), torch.ones(2)),
            (torch.tensor([float("nan"), 0]), torch.ones(2)),
            (torch.zeros(2), torch.tensor([float("inf"), 1])),
        ]:
            with (
                self.subTest(low=low, high=high),
                self.assertRaises((ValueError, TypeError)),
            ):
                Actor(3, 2, (), "relu", low, high)

    def test_critic_retains_gradient_to_actor_actions(self):
        critic = Critic(3, 2, (), "relu")
        with torch.no_grad():
            critic.head.weight.zero_()
            critic.head.weight[0, -2:] = torch.tensor([2.0, 3.0])
        for param in critic.parameters():
            param.requires_grad_(False)
        actions = torch.zeros(4, 2, requires_grad=True)
        critic(torch.zeros(4, 3), actions).sum().backward()
        torch.testing.assert_close(
            actions.grad, torch.tensor([[2.0, 3.0]]).expand(4, 2)
        )
        self.assertTrue(all(p.grad is None for p in critic.parameters()))


if __name__ == "__main__":
    unittest.main()
