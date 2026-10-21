"""DQN agent tests.

Examples:
    >>> # Run with pytest
    >>> # python -m pytest tests/deepqn/test_agent.py
    >>> # Run with unittest
    >>> # python -m unittest tests.deepqn.test_agent
"""

import unittest
from unittest import mock

import numpy as np
import torch
from gymnasium import spaces

from deepQN.dqn.agent import DQNAgent


def _make_agent(
    obs_dim: int = 4,
    act_dim: int = 3,
    hidden_sizes: list[int] | None = None,
    double_dqn: bool = False,
    target_update_interval: int = 1,
    gamma: float = 0.99,
) -> DQNAgent:
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Discrete(act_dim)
    return DQNAgent(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=hidden_sizes or [],
        activation="relu",
        lr=1e-3,
        gamma=gamma,
        target_update_interval=target_update_interval,
        max_grad_norm=1.0,
        double_dqn=double_dqn,
        device="cpu",
    )


class TestDQNAgent(unittest.TestCase):
    def test_act_deterministic_uses_argmax(self) -> None:
        agent = _make_agent(obs_dim=2, act_dim=3, hidden_sizes=[])
        with torch.no_grad():
            layer = agent.q_network.model[0]
            layer.weight.zero_()
            layer.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

        obs = np.array([0.5, -0.2], dtype=np.float32)
        action = agent.act(obs, epsilon=0.0, deterministic=True)

        self.assertEqual(action, 2)
        self.assertEqual(agent.step_count, 0)

    def test_act_epsilon_sampling_uses_action_space(self) -> None:
        agent = _make_agent(obs_dim=4, act_dim=3, hidden_sizes=[])
        obs = np.zeros(4, dtype=np.float32)

        with mock.patch.object(agent.act_space, "sample", return_value=1) as sampler:
            action = agent.act(obs, epsilon=1.0, deterministic=False)

        self.assertEqual(action, 1)
        sampler.assert_called_once()
        self.assertEqual(agent.step_count, 1)

    def test_update_changes_parameters_and_syncs_target(self) -> None:
        torch.manual_seed(0)
        agent = _make_agent(obs_dim=4, act_dim=2, hidden_sizes=[8], double_dqn=True)
        batch_size = 6
        batch = {
            "obs": torch.randn((batch_size, 4), dtype=torch.float32),
            "actions": torch.randint(0, 2, (batch_size,), dtype=torch.int64),
            "rewards": torch.randn((batch_size,), dtype=torch.float32),
            "next_obs": torch.randn((batch_size, 4), dtype=torch.float32),
            "dones": torch.zeros((batch_size,), dtype=torch.float32),
        }

        before = [p.detach().clone() for p in agent.q_network.parameters()]
        stats = agent.update(batch)

        changed = any(
            not torch.allclose(b, a)
            for b, a in zip(before, agent.q_network.parameters())
        )
        self.assertTrue(changed)
        self.assertEqual(agent.train_steps, 1)
        self.assertTrue(np.isfinite(stats.loss))
        self.assertTrue(np.isfinite(stats.td_error))
        for q_param, target_param in zip(
            agent.q_network.parameters(), agent.target_q_network.parameters()
        ):
            self.assertTrue(torch.allclose(q_param, target_param))

    def test_state_dict_round_trip(self) -> None:
        torch.manual_seed(1)
        agent = _make_agent(obs_dim=3, act_dim=2, hidden_sizes=[8])
        agent.act(np.zeros(3, dtype=np.float32), epsilon=0.0, deterministic=True)
        batch = {
            "obs": torch.randn((4, 3), dtype=torch.float32),
            "actions": torch.randint(0, 2, (4,), dtype=torch.int64),
            "rewards": torch.randn((4,), dtype=torch.float32),
            "next_obs": torch.randn((4, 3), dtype=torch.float32),
            "dones": torch.zeros((4,), dtype=torch.float32),
        }
        agent.update(batch)

        state = agent.state_dict()
        restored = _make_agent(obs_dim=3, act_dim=2, hidden_sizes=[8])
        restored.load_state_dict(state)

        self.assertEqual(restored.step_count, agent.step_count)
        self.assertEqual(restored.train_steps, agent.train_steps)
        for param, restored_param in zip(
            agent.q_network.parameters(), restored.q_network.parameters()
        ):
            self.assertTrue(torch.allclose(param, restored_param))
        for q_param, target_param in zip(
            restored.q_network.parameters(), restored.target_q_network.parameters()
        ):
            self.assertTrue(torch.allclose(q_param, target_param))


class TestDQNContracts(unittest.TestCase):
    def _batch(self):
        return {
            "obs": torch.zeros((2, 2)),
            "next_obs": torch.zeros((2, 2)),
            "actions": torch.tensor([0, 0]),
            "rewards": torch.tensor([1.0, 1.0]),
            "dones": torch.tensor([0.0, 1.0]),
        }

    def test_deterministic_act_and_q_value_are_readonly_and_do_not_consume_rng(self):
        agent = _make_agent(obs_dim=2, act_dim=3)
        with torch.no_grad():
            agent.q_network.model[0].weight.zero_()
            agent.q_network.model[0].bias.copy_(torch.tensor([1.0, 3.0, 3.0]))
        agent.step_count = 17
        obs = np.zeros(2, dtype=np.float32)
        np.random.seed(123)
        before = np.random.get_state()
        with (
            mock.patch("numpy.random.rand", side_effect=AssertionError("RNG used")),
            mock.patch.object(
                agent.act_space, "sample", side_effect=AssertionError("RNG used")
            ),
        ):
            self.assertEqual(
                [agent.act(obs, 1.0, deterministic=True) for _ in range(3)], [1, 1, 1]
            )
            self.assertEqual(agent.q_value(obs), 3.0)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(agent.step_count, 17)
        self.assertEqual(agent.train_steps, 0)

    def test_dqn_and_double_dqn_targets_use_online_selection_target_evaluation_and_terminal_mask(
        self,
    ):
        for double_dqn, targets in ((False, [4.5, 1.0]), (True, [2.0, 1.0])):
            with self.subTest(double_dqn=double_dqn):
                agent = _make_agent(
                    obs_dim=2,
                    act_dim=2,
                    double_dqn=double_dqn,
                    target_update_interval=100,
                    gamma=0.5,
                )
                with torch.no_grad():
                    agent.q_network.model[0].weight.zero_()
                    agent.q_network.model[0].bias.copy_(torch.tensor([1.0, 3.0]))
                    agent.target_q_network.model[0].weight.zero_()
                    agent.target_q_network.model[0].bias.copy_(torch.tensor([7.0, 2.0]))
                stats = agent.update(self._batch())
                targets = np.asarray(targets)
                self.assertAlmostEqual(stats.target_value, float(targets.mean()))
                self.assertAlmostEqual(stats.q_value, 1.0)
                self.assertAlmostEqual(
                    stats.td_error, float(np.abs(1.0 - targets).mean())
                )
                self.assertAlmostEqual(stats.loss, float(((1.0 - targets) ** 2).mean()))
                self.assertTrue(
                    all(np.isfinite(value) for value in vars(stats).values())
                )
                self.assertTrue(
                    all(
                        parameter.grad is None
                        for parameter in agent.target_q_network.parameters()
                    )
                )

    def test_target_is_fixed_until_gradient_update_interval_not_action_count(self):
        agent = _make_agent(obs_dim=2, act_dim=2, target_update_interval=3)
        before = [
            parameter.detach().clone()
            for parameter in agent.target_q_network.parameters()
        ]
        for _ in range(10):
            agent.act(np.zeros(2, dtype=np.float32), 0.0)
        for update in (1, 2):
            agent.update(self._batch())
            self.assertEqual(agent.train_steps, update)
            for expected, actual in zip(before, agent.target_q_network.parameters()):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                self.assertIsNone(actual.grad)
        agent.update(self._batch())
        self.assertEqual(agent.train_steps, 3)
        for online, target in zip(
            agent.q_network.parameters(), agent.target_q_network.parameters()
        ):
            torch.testing.assert_close(online, target, rtol=0, atol=0)

    def test_state_dict_is_independent_snapshot(self):
        agent = _make_agent(obs_dim=2, act_dim=2, target_update_interval=3)
        agent.update(self._batch())
        state = agent.state_dict()
        saved = {key: tensor.clone() for key, tensor in state["q_network"].items()}
        agent.update(self._batch())
        for key in saved:
            torch.testing.assert_close(
                state["q_network"][key], saved[key], rtol=0, atol=0
            )
        self.assertEqual(state["train_steps"], 1)

    def test_legacy_state_missing_target_falls_back_to_online(self):
        agent = _make_agent(obs_dim=2, act_dim=2)
        state = agent.state_dict()
        del state["target_q_network"]
        restored = _make_agent(obs_dim=2, act_dim=2)
        restored.load_state_dict(state)
        for online, target in zip(
            restored.q_network.parameters(), restored.target_q_network.parameters()
        ):
            torch.testing.assert_close(online, target, rtol=0, atol=0)

    def test_invalid_counters_and_optimizer_state_are_not_silently_coerced_or_ignored(
        self,
    ):
        for counter in ("step_count", "train_steps"):
            for value in (-1, 1.5, True, "bad"):
                with self.subTest(counter=counter, value=value):
                    agent = _make_agent(obs_dim=2, act_dim=2)
                    state = agent.state_dict()
                    state[counter] = value
                    with self.assertRaises((ValueError, TypeError)):
                        agent.load_state_dict(state)
        agent = _make_agent(obs_dim=2, act_dim=2)
        state = agent.state_dict()
        state["optimizer"] = {"state": {}, "param_groups": []}
        with self.assertRaises(ValueError):
            agent.load_state_dict(state)

    def test_wrong_network_shape_and_nonfinite_observations_are_rejected(self):
        agent = _make_agent(obs_dim=2, act_dim=2)
        state = agent.state_dict()
        state["q_network"]["model.0.weight"] = torch.zeros((3, 2))
        with self.assertRaises((RuntimeError, ValueError)):
            agent.load_state_dict(state)
        for obs in (np.zeros(3), np.array([np.nan, 0.0]), np.array([np.inf, 0.0])):
            with self.subTest(obs=obs), self.assertRaises((ValueError, TypeError)):
                agent.act(obs, 0.0, deterministic=True)

    def test_invalid_epsilon_and_zero_based_action_space_are_validated(self):
        agent = _make_agent(obs_dim=2, act_dim=2)
        for epsilon in (-0.1, 1.1, np.nan):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                agent.act(np.zeros(2), epsilon)
        self.assertEqual(agent.step_count, 0)
        with self.assertRaises(ValueError):
            DQNAgent(
                spaces.Box(-1.0, 1.0, (2,), dtype=np.float32),
                spaces.Discrete(2, start=1),
                [],
                "relu",
                0.001,
                0.5,
                3,
                1.0,
                True,
                "cpu",
            )

    def test_invalid_batch_actions_shapes_rewards_and_done_masks_are_rejected(self):
        for key, value in (
            ("actions", torch.tensor([-1, 0])),
            ("actions", torch.tensor([2, 0])),
            ("actions", torch.tensor([0.5, 0.0])),
            ("obs", torch.zeros((2, 3))),
            ("rewards", torch.tensor([np.nan, 1.0])),
            ("dones", torch.tensor([0.5, 0.0])),
        ):
            with self.subTest(key=key, value=value):
                agent = _make_agent(obs_dim=2, act_dim=2)
                batch = self._batch()
                batch[key] = value
                with self.assertRaises((ValueError, TypeError, RuntimeError)):
                    agent.update(batch)


class TestObservationNormalization(unittest.TestCase):
    def agent(self, normalized=True, low=None, high=None):
        low = (
            np.array([-2.0, 0.0], dtype=np.float32)
            if low is None
            else np.asarray(low, dtype=np.float32)
        )
        high = (
            np.array([2.0, 8.0], dtype=np.float32)
            if high is None
            else np.asarray(high, dtype=np.float32)
        )
        result = DQNAgent(
            spaces.Box(low, high, dtype=np.float32),
            spaces.Discrete(2),
            [],
            "relu",
            0.001,
            0.5,
            100,
            1.0,
            False,
            "cpu",
            normalize_observations=normalized,
        )
        with torch.no_grad():
            for network in (result.q_network, result.target_q_network):
                network.model[0].weight.copy_(torch.eye(2))
                network.model[0].bias.zero_()
        return result

    def test_act_and_q_value_encode_asymmetric_bounds_while_raw_control_is_unchanged(
        self,
    ):
        normalized, raw = self.agent(), self.agent(normalized=False)
        obs = np.array([2.0, 5.0], dtype=np.float32)
        # Center=[0,4], half-range=[2,4], so encoded obs=[1,0.25].
        self.assertEqual(normalized.act(obs, 0.0, deterministic=True), 0)
        self.assertEqual(normalized.q_value(obs), 1.0)
        self.assertEqual(raw.act(obs, 0.0, deterministic=True), 1)
        self.assertEqual(raw.q_value(obs), 5.0)
        self.assertEqual(normalized.step_count, 0)
        np.testing.assert_array_equal(obs, [2.0, 5.0])

    def test_update_encodes_both_observations_consistently_without_mutating_batch(self):
        agent = self.agent()
        batch = {
            "obs": torch.tensor([[2.0, 5.0]]),
            "next_obs": torch.tensor([[0.0, 8.0]]),
            "actions": torch.tensor([0]),
            "rewards": torch.tensor([-1.0]),
            "dones": torch.tensor([0.0]),
        }
        before = {key: value.clone() for key, value in batch.items()}
        # Online Q(action0)=1; encoded next=[0,1], target=-1+0.5*1=-0.5.
        stats = agent.update(batch)
        self.assertEqual(stats.q_value, 1.0)
        self.assertEqual(stats.target_value, -0.5)
        self.assertEqual(stats.td_error, 1.5)
        self.assertEqual(stats.loss, 2.25)
        for key in batch:
            torch.testing.assert_close(batch[key], before[key], rtol=0, atol=0)

    def test_normalization_rejects_invalid_bounds_and_handles_extreme_finite_ranges(
        self,
    ):
        for low, high in (
            ([-np.inf, 0.0], [2.0, 8.0]),
            ([-2.0, 0.0], [np.inf, 8.0]),
            ([0.0, 0.0], [0.0, 8.0]),
        ):
            with self.subTest(low=low, high=high), self.assertRaises(ValueError):
                self.agent(low=low, high=high)
        extreme = self.agent(low=[-3e38, -2e38], high=[3e38, 2e38])
        self.assertAlmostEqual(
            extreme.q_value(np.array([3e38, 2e38], dtype=np.float32)), 1.0
        )
        self.assertTrue(np.isfinite(extreme.q_value(np.zeros(2, dtype=np.float32))))

    def test_normalized_state_round_trip_is_independent_and_survives_device_move(self):
        original = self.agent()
        original.step_count = 17
        state = original.state_dict()
        self.assertIs(state["normalize_observations"], True)
        self.assertEqual(state["observation_low"], [-2.0, 0.0])
        self.assertEqual(state["observation_high"], [2.0, 8.0])
        restored = self.agent()
        restored.load_state_dict(state)
        self.assertEqual(restored.step_count, 17)
        obs = np.array([2.0, 5.0], dtype=np.float32)
        self.assertEqual(restored.q_value(obs), original.q_value(obs))
        self.assertIs(restored.to("cpu"), restored)
        self.assertEqual(restored.q_value(obs), 1.0)
        # A saved snapshot must not alias weights or normalization tensors.
        with torch.no_grad():
            original.q_network.model[0].bias.fill_(99.0)
            original._obs_center.fill_(99.0)
            original._observation_low.fill_(-99.0)
        self.assertEqual(restored.q_value(obs), 1.0)
        self.assertEqual(state["q_network"]["model.0.bias"].tolist(), [0.0, 0.0])
        self.assertEqual(state["observation_low"], [-2.0, 0.0])
        for value in vars(restored).values():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device.type, "cpu")

    def test_incompatible_normalization_or_bounds_rejected_before_weight_restore(self):
        state = self.agent().state_dict()
        for destination in (
            self.agent(normalized=False),
            self.agent(low=[-3.0, 0.0], high=[3.0, 8.0]),
        ):
            with self.subTest(
                normalized=destination.state_dict()["normalize_observations"]
            ):
                with torch.no_grad():
                    destination.q_network.model[0].bias.fill_(-5.0)
                before = {
                    key: value.clone()
                    for key, value in destination.q_network.state_dict().items()
                }
                with self.assertRaises(ValueError):
                    destination.load_state_dict(state)
                for key, expected in before.items():
                    torch.testing.assert_close(
                        destination.q_network.state_dict()[key],
                        expected,
                        rtol=0,
                        atol=0,
                    )

    def test_legacy_missing_normalization_flag_loads_only_into_raw_agent(self):
        legacy = self.agent(normalized=False).state_dict()
        del legacy["normalize_observations"]
        raw = self.agent(normalized=False)
        raw.load_state_dict(legacy)
        self.assertEqual(raw.q_value(np.array([2.0, 5.0], dtype=np.float32)), 5.0)
        with self.assertRaises(ValueError):
            self.agent().load_state_dict(legacy)


if __name__ == "__main__":
    unittest.main()
