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


if __name__ == "__main__":
    unittest.main()
