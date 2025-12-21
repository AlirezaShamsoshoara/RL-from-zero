import unittest
import numpy as np

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


if __name__ == "__main__":
    unittest.main()
