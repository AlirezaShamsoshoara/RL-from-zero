"""DQN network tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/deepqn/test_networks.py
    >>> # Run with unittest
    >>> # python -m unittest tests.deepqn.test_networks
"""

import unittest

import torch

from deepQN.dqn.networks import QNetwork, _get_activation


class TestNetworks(unittest.TestCase):
    def test_get_activation_variants(self) -> None:
        self.assertIsInstance(_get_activation("tanh"), torch.nn.Tanh)
        self.assertIsInstance(_get_activation("sigmoid"), torch.nn.Sigmoid)
        self.assertIsInstance(_get_activation("gelu"), torch.nn.GELU)
        self.assertIsInstance(_get_activation("relu"), torch.nn.ReLU)
        self.assertIsInstance(_get_activation("unknown"), torch.nn.ReLU)

    def test_qnetwork_forward_shape(self) -> None:
        net = QNetwork(obs_dim=4, action_dim=3, hidden_sizes=[8], activation="relu")
        obs = torch.zeros((5, 4), dtype=torch.float32)

        out = net(obs)

        self.assertEqual(out.shape, (5, 3))
        self.assertEqual(out.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
