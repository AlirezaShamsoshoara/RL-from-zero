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
        with self.assertRaises(ValueError):
            get_activation("unknown")

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

    def test_deterministic_act_is_argmax_without_sampling_or_rng_change(self) -> None:
        from unittest.mock import patch

        model = ActorCritic(obs_dim=6, act_dim=3, hidden_sizes=(8,))
        obs = torch.ones(4, 6)
        logits, _ = model(obs)
        rng_before = torch.random.get_rng_state().clone()
        with patch(
            "torch.distributions.Categorical.sample",
            side_effect=AssertionError("sampled"),
        ):
            action, log_prob, entropy, value = model.act(obs, deterministic=True)
        self.assertTrue(torch.equal(action, logits.argmax(dim=-1)))
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        for tensor in (log_prob, entropy, value):
            self.assertEqual(tensor.shape, (4,))
            self.assertTrue(torch.isfinite(tensor).all())

    def test_invalid_network_dimensions_and_hidden_sizes_are_rejected(self) -> None:
        for kwargs in (
            {"obs_dim": 0},
            {"obs_dim": True},
            {"obs_dim": 1.5},
            {"act_dim": 0},
            {"act_dim": False},
            {"hidden_sizes": [0]},
            {"hidden_sizes": [True]},
            {"hidden_sizes": [2.5]},
            {"activation": "unsupported"},
        ):
            with self.subTest(kwargs=kwargs):
                arguments = {"obs_dim": 6, "act_dim": 3, "hidden_sizes": (8,)}
                arguments.update(kwargs)
                with self.assertRaises(ValueError):
                    ActorCritic(**arguments)

    def test_forward_rejects_wrong_shape_and_nonfinite_observations(self) -> None:
        model = ActorCritic(obs_dim=6, act_dim=3, hidden_sizes=(8,))
        for obs in (
            torch.ones(5),
            torch.ones(2, 3, 6),
            torch.full((2, 6), float("nan")),
        ):
            with self.subTest(shape=tuple(obs.shape)):
                with self.assertRaises(ValueError):
                    model(obs)

    def test_evaluate_actions_rejects_wrong_batch_and_invalid_actions(self) -> None:
        model = ActorCritic(obs_dim=6, act_dim=3, hidden_sizes=(8,))
        for actions in (torch.tensor([0]), torch.tensor([0, 3]), torch.tensor([-1, 0])):
            with self.subTest(actions=actions.tolist()):
                with self.assertRaises(ValueError):
                    model.evaluate_actions(torch.ones(2, 6), actions)


if __name__ == "__main__":
    unittest.main()
