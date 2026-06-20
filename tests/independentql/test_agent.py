"""Independent Q-learning agent tests.

Examples:
    >>> # python -m pytest tests/independentql/test_agent.py
"""

import unittest

import numpy as np

from independent_ql.agent import IndependentQLearningAgent, Transition


def _make_agent(n_agents=2, n_states=10, n_actions=3, eps_start=1.0, eps_end=0.05):
    return IndependentQLearningAgent(
        n_agents=n_agents,
        n_states=n_states,
        n_actions=n_actions,
        alpha=0.5,
        gamma=0.95,
        epsilon_start=eps_start,
        epsilon_end=eps_end,
        epsilon_decay=0.01,
    )


class TestIndependentQLearningAgent(unittest.TestCase):
    def test_act_returns_valid_actions(self) -> None:
        np.random.seed(0)
        agent = _make_agent()
        actions = agent.act([0, 1])
        self.assertEqual(len(actions), 2)
        for a in actions:
            self.assertTrue(0 <= a < agent.n_actions)

    def test_act_validates_inputs(self) -> None:
        agent = _make_agent()
        with self.assertRaises(ValueError):
            agent.act([0])  # wrong number of states
        with self.assertRaises(ValueError):
            agent.act([0, 999])  # state out of bounds

    def test_epsilon_decays(self) -> None:
        agent = _make_agent(eps_start=1.0, eps_end=0.05)
        first = agent.epsilon()
        for _ in range(500):
            agent.act([0, 0])
        later = agent.epsilon()
        self.assertLess(later, first)
        self.assertGreaterEqual(later, 0.05 - 1e-9)

    def test_greedy_actions_pick_argmax(self) -> None:
        agent = _make_agent()
        agent.Q[0, 3] = np.array([0.1, 0.9, 0.2], dtype=np.float32)
        agent.Q[1, 4] = np.array([0.5, 0.2, 0.1], dtype=np.float32)
        self.assertEqual(agent.greedy_actions([3, 4]), [1, 0])

    def test_update_terminal_and_bootstrap(self) -> None:
        agent = _make_agent(n_agents=1, n_states=5, n_actions=2)
        # Terminal transition: target == reward, so Q moves toward reward.
        agent.update([Transition(agent=0, state=0, action=1, reward=1.0,
                                 next_state=1, done=True)])
        self.assertAlmostEqual(agent.Q[0, 0, 1], 0.5 * 1.0, places=5)  # alpha=0.5 from 0

        # Bootstrap: target = reward + gamma * max_a Q(next).
        agent.Q[0, 2] = np.array([0.0, 2.0], dtype=np.float32)
        agent.update([Transition(agent=0, state=3, action=0, reward=0.0,
                                 next_state=2, done=False)])
        expected = 0.5 * (0.0 + 0.95 * 2.0)  # from Q=0, alpha=0.5
        self.assertAlmostEqual(agent.Q[0, 3, 0], expected, places=5)

    def test_q_table_shape(self) -> None:
        agent = _make_agent(n_agents=3, n_states=8, n_actions=4)
        self.assertEqual(agent.Q.shape, (3, 8, 4))


if __name__ == "__main__":
    unittest.main()
