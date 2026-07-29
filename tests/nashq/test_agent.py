"""Nash Q-learning agent tests.

Examples:
    >>> # python -m pytest tests/nashq/test_agent.py
"""

import unittest

import numpy as np

from nash_ql.agent import NashQLearningAgent, Transition


def _make_agent(n_states=20, n_actions=3, eps_start=1.0, eps_end=0.05):
    return NashQLearningAgent(
        n_agents=2, n_states=n_states, n_actions=n_actions,
        alpha=0.5, gamma=0.95,
        epsilon_start=eps_start, epsilon_end=eps_end, epsilon_decay=0.01,
    )


class TestNashQLearningAgent(unittest.TestCase):
    def test_requires_two_agents(self) -> None:
        with self.assertRaises(ValueError):
            NashQLearningAgent(n_agents=3, n_states=10, n_actions=3,
                               alpha=0.1, gamma=0.95,
                               epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.01)

    def test_q_table_shape(self) -> None:
        agent = _make_agent(n_states=12, n_actions=3)
        self.assertEqual(agent.Q.shape, (2, 12, 3, 3))

    def test_act_returns_two_valid_actions(self) -> None:
        np.random.seed(0)
        agent = _make_agent()
        actions = agent.act([0, 0])
        self.assertEqual(len(actions), 2)
        for a in actions:
            self.assertTrue(0 <= a < agent.n_actions)

    def test_act_validates_inputs(self) -> None:
        agent = _make_agent()
        with self.assertRaises(ValueError):
            agent.act([0])  # wrong count
        with self.assertRaises(ValueError):
            agent.act([0, 999])  # out of bounds

    def test_epsilon_decays(self) -> None:
        agent = _make_agent()
        first = agent.epsilon()
        for _ in range(500):
            agent.act([0, 0])
        self.assertLess(agent.epsilon(), first)

    def test_greedy_actions_in_range(self) -> None:
        np.random.seed(0)
        agent = _make_agent()
        agent.Q[0, 5] = np.eye(3, dtype=np.float32)
        agent.Q[1, 5] = np.eye(3, dtype=np.float32)
        acts = agent.greedy_actions([5, 5])
        self.assertEqual(len(acts), 2)
        for a in acts:
            self.assertTrue(0 <= a < 3)

    def test_update_terminal_moves_toward_reward(self) -> None:
        agent = _make_agent(n_states=10, n_actions=3)
        agent.update([Transition(agent=0, state=2, joint_action=(1, 2), reward=1.0,
                                 next_state=3, done=True)])
        # alpha=0.5 from 0 with terminal target=reward -> 0.5
        self.assertAlmostEqual(agent.Q[0, 2, 1, 2], 0.5, places=5)

    def test_update_bootstrap_is_finite(self) -> None:
        agent = _make_agent(n_states=10, n_actions=3)
        agent.Q[0, 4] = np.ones((3, 3), dtype=np.float32)
        agent.Q[1, 4] = np.ones((3, 3), dtype=np.float32)
        before = agent.Q[0, 1, 0, 0]
        agent.update([Transition(agent=0, state=1, joint_action=(0, 0), reward=0.0,
                                 next_state=4, done=False)])
        self.assertTrue(np.isfinite(agent.Q[0, 1, 0, 0]))
        self.assertNotEqual(agent.Q[0, 1, 0, 0], before)

    def test_update_validates_joint_action_size(self) -> None:
        agent = _make_agent()
        with self.assertRaises(ValueError):
            agent.update([Transition(agent=0, state=0, joint_action=(1,), reward=0.0,
                                     next_state=1, done=False)])

    def test_best_response_action(self) -> None:
        agent = _make_agent(n_states=10, n_actions=3)
        # Agent 0 is axis 0: action 2 has the highest expected value over agent 1.
        agent.Q[0, 5] = np.array([[0.0, 0.0, 0.0],
                                  [0.1, 0.1, 0.1],
                                  [0.9, 0.8, 0.7]], dtype=np.float32)
        self.assertEqual(agent.best_response_action(5, agent=0), 2)
        # Agent 1 is axis 1: column 1 is best on average.
        agent.Q[1, 6] = np.array([[0.0, 0.9, 0.0],
                                  [0.0, 0.8, 0.1],
                                  [0.0, 0.7, 0.0]], dtype=np.float32)
        self.assertEqual(agent.best_response_action(6, agent=1), 1)


if __name__ == "__main__":
    unittest.main()
