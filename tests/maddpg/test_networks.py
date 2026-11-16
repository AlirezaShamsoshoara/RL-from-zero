"""MADDPG network tests.

Examples:
    >>> # python -m pytest tests/maddpg/test_networks.py
"""

import unittest

import torch

from MADDPG.maddpg.networks import Actor, Critic, build_mlp


class TestBuildMLP(unittest.TestCase):
    def test_shape(self) -> None:
        mlp = build_mlp(4, [8, 8], "relu")
        y = mlp(torch.zeros(3, 4))
        self.assertEqual(y.shape, (3, 8))

    def test_activation_choices(self) -> None:
        for act in ["relu", "tanh", "leaky_relu", "elu"]:
            mlp = build_mlp(2, [4], act)
            self.assertEqual(mlp(torch.zeros(1, 2)).shape, (1, 4))

    def test_unknown_activation_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_mlp(2, [4], "banana")


class TestActor(unittest.TestCase):
    def test_output_shape_and_bounds(self) -> None:
        low = torch.tensor([0.0, -1.0, 2.0])
        high = torch.tensor([1.0, 1.0, 4.0])
        actor = Actor(obs_dim=5, act_dim=3, hidden_sizes=[8, 8],
                      activation="relu", action_low=low, action_high=high)
        obs = torch.randn(7, 5) * 3.0
        with torch.no_grad():
            a = actor(obs)
        self.assertEqual(a.shape, (7, 3))
        # Tanh + affine scale must keep every dim inside [low, high].
        self.assertTrue(torch.all(a >= low).item())
        self.assertTrue(torch.all(a <= high).item())


class TestCritic(unittest.TestCase):
    def test_output_shape(self) -> None:
        critic = Critic(total_obs_dim=6, total_act_dim=4, hidden_sizes=[8], activation="relu")
        q = critic(torch.zeros(5, 6), torch.zeros(5, 4))
        self.assertEqual(q.shape, (5, 1))

    def test_gradient_flows_to_both_inputs(self) -> None:
        critic = Critic(total_obs_dim=3, total_act_dim=2, hidden_sizes=[4], activation="relu")
        obs = torch.zeros(2, 3, requires_grad=True)
        act = torch.zeros(2, 2, requires_grad=True)
        critic(obs, act).sum().backward()
        self.assertIsNotNone(obs.grad)
        self.assertIsNotNone(act.grad)


if __name__ == "__main__":
    unittest.main()
