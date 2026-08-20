"""Nash-QL LineWorld environment tests.

Examples:
    >>> # python -m pytest tests/nashq/test_env.py
"""

import unittest

import numpy as np

from nash_ql.envs.line_world import LineWorldEnv


class TestLineWorld(unittest.TestCase):
    def test_shared_state_across_agents(self) -> None:
        # Nash Q-learning needs both agents to observe the SAME joint state index.
        env = LineWorldEnv(n_agents=2, grid_length=7)
        states = env.reset(seed=0)
        self.assertEqual(states[0], states[1])
        self.assertEqual(env.n_states, 7 ** 2)
        self.assertEqual(env.n_actions, 3)

    def test_move_right(self) -> None:
        env = LineWorldEnv(n_agents=1, grid_length=5, goal_positions=[4])
        env.reset(seed=0)
        env.positions = [0]
        env.step([1])
        self.assertEqual(env.positions[0], 1)

    def test_goal_reward_once_then_freeze(self) -> None:
        env = LineWorldEnv(n_agents=1, grid_length=5, goal_positions=[1],
                           step_penalty=-0.02, goal_reward=1.0, shared_goal_bonus=0.0,
                           max_steps=10)
        env.reset(seed=0)
        env.positions = [0]
        res1 = env.step([1])  # reach goal
        self.assertAlmostEqual(res1.rewards[0], -0.02 + 1.0, places=5)
        self.assertTrue(res1.terminated[0])
        res2 = env.step([1])  # frozen, no further reward
        self.assertAlmostEqual(res2.rewards[0], 0.0, places=5)
        self.assertEqual(env.positions[0], 1)

    def test_shared_bonus_when_all_reach(self) -> None:
        env = LineWorldEnv(n_agents=2, grid_length=4, goal_positions=[1, 2],
                           step_penalty=0.0, goal_reward=1.0, shared_goal_bonus=0.5,
                           collision_penalty=0.0)
        env.reset(seed=0)
        env.positions = [0, 1]
        res = env.step([1, 1])
        self.assertTrue(all(res.terminated))
        for r in res.rewards:
            self.assertAlmostEqual(r, 1.5, places=5)

    def test_collision_penalty(self) -> None:
        env = LineWorldEnv(n_agents=2, grid_length=5, goal_positions=[4, 3],
                           step_penalty=0.0, goal_reward=0.0, shared_goal_bonus=0.0,
                           collision_penalty=-0.1)
        env.reset(seed=0)
        env.positions = [1, 3]
        res = env.step([1, 2])  # both move to cell 2
        self.assertAlmostEqual(res.rewards[0], -0.1, places=5)
        self.assertAlmostEqual(res.rewards[1], -0.1, places=5)

    def test_truncation(self) -> None:
        env = LineWorldEnv(n_agents=1, grid_length=5, goal_positions=[4], max_steps=3)
        env.reset(seed=0)
        env.positions = [0]
        res = None
        for _ in range(3):
            res = env.step([0])
        self.assertTrue(res.truncated)

    def test_render_rgb(self) -> None:
        env = LineWorldEnv(n_agents=2, grid_length=6)
        env.reset(seed=0)
        frame = env.render()
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)
        self.assertEqual(frame.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
