import unittest
from dataclasses import is_dataclass
from unittest.mock import patch

import numpy as np
import torch

from Qlearning.ql.agent import QLearningAgent, Transition

# To run: `python -m pytest tests/qlearning/test_agent.py`


class TestQLearningAgent(unittest.TestCase):
    def test_epsilon_decay_respects_bounds(self) -> None:
        agent = QLearningAgent(
            n_states=2,
            n_actions=2,
            alpha=0.1,
            gamma=0.9,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.5,
        )

        agent.global_step = 0
        self.assertAlmostEqual(agent.epsilon(), agent.eps_start)

        agent.global_step = 5
        expected = agent.eps_end + (agent.eps_start - agent.eps_end) * np.exp(
            -agent.eps_decay * agent.global_step
        )
        self.assertAlmostEqual(agent.epsilon(), expected)
        self.assertGreater(agent.epsilon(), agent.eps_end)

        agent.global_step = 1_000_000
        self.assertAlmostEqual(agent.epsilon(), agent.eps_end, delta=1e-4)

    def test_act_increments_step_and_returns_greedy_argmax(self) -> None:
        agent = QLearningAgent(
            n_states=3,
            n_actions=3,
            alpha=0.1,
            gamma=0.9,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
        )
        agent.Q[2] = np.array([0.1, 0.6, 0.3], dtype=np.float32)

        self.assertEqual(agent.global_step, 0)
        np.random.seed(123)
        action = agent.act(2)

        self.assertEqual(action, 1)
        self.assertEqual(agent.global_step, 1)

    def test_act_breaks_ties_across_max_actions(self) -> None:
        agent = QLearningAgent(
            n_states=1,
            n_actions=3,
            alpha=0.1,
            gamma=0.9,
            epsilon_start=0.0,
            epsilon_end=0.0,
            epsilon_decay=1.0,
        )
        agent.Q[0] = np.array([1.0, 1.0, 0.2], dtype=np.float32)

        np.random.seed(0)
        actions = [agent.act(0) for _ in range(6)]

        self.assertTrue(set(actions).issubset({0, 1}))
        self.assertGreaterEqual(len(set(actions)), 2)

    def test_update_applies_td_error(self) -> None:
        agent = QLearningAgent(
            n_states=3,
            n_actions=2,
            alpha=0.5,
            gamma=0.9,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.1,
        )
        agent.Q[1, 0] = 0.4
        agent.Q[2] = np.array([0.6, 0.8], dtype=np.float32)

        transition = Transition(s=1, a=0, r=1.0, s2=2, done=False)
        target = transition.r + agent.gamma * np.max(agent.Q[transition.s2])
        expected = agent.Q[1, 0] + agent.alpha * (target - agent.Q[1, 0])

        agent.update(transition)

        self.assertAlmostEqual(agent.Q[1, 0], expected, places=6)

    def test_update_ignores_bootstrap_when_done(self) -> None:
        agent = QLearningAgent(
            n_states=2,
            n_actions=2,
            alpha=0.25,
            gamma=0.9,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=0.1,
        )
        agent.Q[0, 1] = 0.5
        transition = Transition(s=0, a=1, r=1.2, s2=1, done=True)
        expected = agent.Q[0, 1] + agent.alpha * (transition.r - agent.Q[0, 1])

        agent.update(transition)

        self.assertAlmostEqual(agent.Q[0, 1], expected)


class TestAgentContracts(unittest.TestCase):
    def _agent(self, **overrides):
        kwargs = dict(
            n_states=3,
            n_actions=3,
            alpha=0.5,
            gamma=0.9,
            epsilon_start=0.8,
            epsilon_end=0.1,
            epsilon_decay=0.2,
        )
        kwargs.update(overrides)
        return QLearningAgent(**kwargs)

    def test_deterministic_actions_are_first_argmax_without_rng_or_counter_mutation(
        self,
    ):
        agent = self._agent(epsilon_start=1.0, epsilon_end=1.0)
        agent.Q[1] = [2.0, 2.0, -1.0]
        agent.global_step = 17
        np.random.seed(123)
        before = np.random.get_state()
        with (
            patch("numpy.random.rand", side_effect=AssertionError("RNG used")),
            patch("numpy.random.randint", side_effect=AssertionError("RNG used")),
            patch("numpy.random.choice", side_effect=AssertionError("RNG used")),
        ):
            actions = [agent.act(1, deterministic=True) for _ in range(4)]
        after = np.random.get_state()
        self.assertEqual(actions, [0, 0, 0, 0])
        self.assertEqual(agent.global_step, 17)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_exploration_uses_random_action_and_increments_once(self):
        agent = self._agent(epsilon_start=1.0, epsilon_end=1.0)
        with (
            patch("numpy.random.rand", return_value=0.0),
            patch("numpy.random.randint", return_value=2) as randint,
        ):
            action = agent.act(0)
        self.assertEqual(action, 2)
        randint.assert_called_once_with(3)
        self.assertEqual(agent.global_step, 1)

    def test_update_returns_finite_dataclass_and_changes_only_selected_entry(self):
        agent = self._agent()
        agent.Q[0, 1] = 0.4
        agent.Q[2] = [0.0, 1.0, 0.2]
        before = agent.Q.copy()
        stats = agent.update(Transition(0, 1, 1.0, 2, False))
        self.assertTrue(is_dataclass(stats))
        self.assertAlmostEqual(stats.target, 1.9, places=6)
        self.assertAlmostEqual(stats.td_error, 1.5, places=6)
        self.assertAlmostEqual(stats.q_value, 1.15, places=6)
        for value in (stats.target, stats.td_error, stats.q_value):
            self.assertIsInstance(value, float)
            self.assertTrue(np.isfinite(value))
        before[0, 1] = stats.q_value
        np.testing.assert_allclose(agent.Q, before)
        self.assertEqual(agent.global_step, 0)

    def test_terminal_stats_have_reward_only_target(self):
        agent = self._agent()
        agent.Q[2] = [100.0, 200.0, 300.0]
        stats = agent.update(Transition(0, 0, 2.0, 2, True))
        self.assertEqual(stats.target, 2.0)
        self.assertEqual(stats.td_error, 2.0)
        self.assertEqual(stats.q_value, 1.0)

    def test_state_round_trip_copies_table_and_restores_exploration_counter(self):
        agent = self._agent()
        agent.Q[:] = np.arange(9).reshape(3, 3)
        agent.global_step = 29
        state = agent.state_dict()
        self.assertEqual(set(state), {"q_table", "global_step"})
        self.assertIsInstance(state["q_table"], np.ndarray)
        self.assertFalse(np.shares_memory(state["q_table"], agent.Q))
        restored = self._agent()
        restored.load_state_dict(state)
        np.testing.assert_array_equal(restored.Q, agent.Q)
        self.assertEqual(restored.global_step, 29)
        self.assertEqual(restored.epsilon(), agent.epsilon())
        self.assertFalse(np.shares_memory(restored.Q, state["q_table"]))
        state["q_table"][:] = -999.0
        self.assertEqual(restored.Q[2, 2], 8.0)
        self.assertEqual(agent.Q[2, 2], 8.0)

    def test_load_state_accepts_cpu_tensor_without_aliasing(self):
        table = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        agent = self._agent()
        agent.load_state_dict({"q_table": table, "global_step": 4})
        np.testing.assert_array_equal(agent.Q, table.numpy())
        table.fill_(-10.0)
        self.assertEqual(agent.Q[2, 2], 8.0)
        self.assertEqual(agent.global_step, 4)

    def test_malformed_state_shape_values_and_step_are_rejected(self):
        invalid = [
            {"q_table": np.zeros((2, 3)), "global_step": 0},
            {"q_table": np.zeros(9), "global_step": 0},
            {"q_table": np.full((3, 3), np.nan), "global_step": 0},
            {"q_table": np.full((3, 3), np.inf), "global_step": 0},
            {"q_table": np.zeros((3, 3)), "global_step": -1},
            {"q_table": np.zeros((3, 3)), "global_step": 1.5},
            {"q_table": np.zeros((3, 3)), "global_step": "bad"},
        ]
        for state in invalid:
            with self.subTest(state=state):
                with self.assertRaises((ValueError, TypeError)):
                    self._agent().load_state_dict(state)

    def test_missing_q_table_is_rejected(self):
        with self.assertRaises(KeyError):
            self._agent().load_state_dict({"global_step": 0})

    def test_overflowing_td_update_is_rejected_without_table_mutation(self):
        agent = self._agent(alpha=1.0)
        before = agent.Q.copy()
        with self.assertRaises(ValueError):
            agent.update(Transition(0, 0, 1e40, 1, False))
        np.testing.assert_array_equal(agent.Q, before)

    def test_invalid_constructor_dimensions_and_hyperparameters_are_rejected(self):
        invalid = [
            {"n_states": 0},
            {"n_actions": -1},
            {"n_states": 1.5},
            {"alpha": 0.0},
            {"alpha": 1.1},
            {"alpha": np.nan},
            {"gamma": -0.1},
            {"gamma": 1.1},
            {"gamma": np.inf},
            {"epsilon_start": 1.1},
            {"epsilon_end": -0.1},
            {"epsilon_start": 0.1, "epsilon_end": 0.2},
            {"epsilon_decay": -0.1},
            {"epsilon_decay": np.nan},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ValueError, TypeError)):
                    self._agent(**kwargs)

    def test_gamma_endpoints_zero_decay_and_alpha_one_are_valid(self):
        for gamma in (0.0, 1.0):
            with self.subTest(gamma=gamma):
                agent = self._agent(
                    gamma=gamma,
                    alpha=1.0,
                    epsilon_decay=0.0,
                    epsilon_start=0.0,
                    epsilon_end=0.0,
                )
                agent.Q[2] = [1.0, 2.0, 3.0]
                stats = agent.update(Transition(0, 0, 1.0, 2, False))
                self.assertEqual(stats.q_value, 1.0 + 3.0 * gamma)
                self.assertEqual(agent.epsilon(), 0.0)

    def test_invalid_action_states_are_rejected_without_counter_mutation(self):
        for state in (-1, 3, 0.5):
            with self.subTest(state=state):
                agent = self._agent()
                for deterministic in (False, True):
                    with self.assertRaises((ValueError, TypeError, IndexError)):
                        agent.act(state, deterministic=deterministic)
                self.assertEqual(agent.global_step, 0)

    def test_invalid_transitions_are_rejected_without_table_mutation(self):
        invalid = [
            Transition(-1, 0, 1.0, 1, False),
            Transition(3, 0, 1.0, 1, False),
            Transition(0, -1, 1.0, 1, False),
            Transition(0, 3, 1.0, 1, False),
            Transition(0, 0, 1.0, 3, False),
            Transition(0, 0, 1.0, -1, False),
            Transition(0, 0, np.nan, 1, False),
            Transition(0, 0, np.inf, 1, False),
        ]
        for transition in invalid:
            with self.subTest(transition=transition):
                agent = self._agent()
                before = agent.Q.copy()
                with self.assertRaises((ValueError, TypeError, IndexError)):
                    agent.update(transition)
                np.testing.assert_array_equal(agent.Q, before)


if __name__ == "__main__":
    unittest.main()
