"""MAPPO network tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/mappo/test_networks.py
    >>> # Run with unittest
    >>> # python -m unittest tests.mappo.test_networks
"""

import unittest

import torch

from MAPPO.mappo.networks import ActorCritic, CentralizedCritic, DecentralizedCritic, get_activation


class TestNetworks(unittest.TestCase):
    def test_get_activation_defaults_to_tanh(self) -> None:
        self.assertIsInstance(get_activation("relu"), torch.nn.ReLU)
        self.assertIsInstance(get_activation(None), torch.nn.Tanh)
        self.assertIsInstance(get_activation("tanh"), torch.nn.Tanh)

    def test_actor_action_evaluate_match(self) -> None:
        torch.manual_seed(0)
        model = ActorCritic(
            obs_dim=4,
            act_dim=3,
            state_dim=8,
            actor_hidden_sizes=(8,),
            critic_hidden_sizes=(8,),
            activation="tanh",
            use_centralized_critic=True,
        )
        obs = torch.randn((6, 4), dtype=torch.float32)

        actions, logprob, entropy = model.act(obs)
        eval_logprob, eval_entropy = model.evaluate_actions(obs, actions)

        self.assertTrue(torch.allclose(logprob, eval_logprob, atol=1e-6))
        self.assertAlmostEqual(float(entropy.mean().item()), float(eval_entropy.mean().item()), places=6)
        self.assertEqual(actions.shape, (6,))

    def test_critic_value_shapes(self) -> None:
        obs = torch.zeros((5, 4), dtype=torch.float32)
        state = torch.zeros((5, 8), dtype=torch.float32)

        centralized = ActorCritic(
            obs_dim=4,
            act_dim=2,
            state_dim=8,
            actor_hidden_sizes=(8,),
            critic_hidden_sizes=(8,),
            activation="relu",
            use_centralized_critic=True,
        )
        decentralized = ActorCritic(
            obs_dim=4,
            act_dim=2,
            state_dim=8,
            actor_hidden_sizes=(8,),
            critic_hidden_sizes=(8,),
            activation="relu",
            use_centralized_critic=False,
        )

        value_c = centralized.get_value(state)
        value_d = decentralized.get_value(obs)

        self.assertIsInstance(centralized.critic, CentralizedCritic)
        self.assertIsInstance(decentralized.critic, DecentralizedCritic)
        self.assertEqual(value_c.shape, (5,))
        self.assertEqual(value_d.shape, (5,))


if __name__ == "__main__":
    unittest.main()
