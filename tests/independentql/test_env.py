"""LineWorld environment tests.

Examples:
    >>> # python -m pytest tests/independentql/test_env.py
"""

import unittest

import numpy as np

from independent_ql.envs.line_world import LineWorldEnv


class TestLineWorld(unittest.TestCase):
    def test_reset_shapes_and_state_bounds(self) -> None:
        env = LineWorldEnv(n_agents=2, grid_length=7)
        states = env.reset(seed=0)
        self.assertEqual(len(states), 2)
        for s in states:
            self.assertTrue(0 <= s < env.n_states)
        self.assertEqual(env.n_actions, 3)

    def test_move_right_and_left_bounds(self) -> None:
        env = LineWorldEnv(n_agents=1, grid_length=5, goal_positions=[4])
        env.reset(seed=0)
        env.positions = [0]
        env.step([1])  # right
        self.assertEqual(env.positions[0], 1)
        env.positions = [0]
        env.step([2])  # left at the wall stays
        self.assertEqual(env.positions[0], 0)

    def test_goal_reward_and_termination(self) -> None:
        env = LineWorldEnv(n_agents=1, grid_length=5, goal_positions=[1],
                           step_penalty=-0.02, goal_reward=1.0, shared_goal_bonus=0.0)
        env.reset(seed=0)
        env.positions = [0]
        res = env.step([1])  # move to cell 1 == goal
        self.assertTrue(res.terminated[0])
        # reward = step_penalty + goal_reward
        self.assertAlmostEqual(res.rewards[0], -0.02 + 1.0, places=5)

    def test_shared_bonus_when_all_reach(self) -> None:
        env = LineWorldEnv(n_agents=2, grid_length=4, goal_positions=[1, 2],
                           step_penalty=0.0, goal_reward=1.0, shared_goal_bonus=0.5,
                           collision_penalty=0.0)
        env.reset(seed=0)
        env.positions = [0, 1]
        res = env.step([1, 1])  # both step right onto their goals
        self.assertTrue(all(res.terminated))
        for r in res.rewards:
            self.assertAlmostEqual(r, 1.0 + 0.5, places=5)

    def test_collision_penalty(self) -> None:
        env = LineWorldEnv(n_agents=2, grid_length=5, goal_positions=[4, 3],
                           step_penalty=0.0, goal_reward=0.0, shared_goal_bonus=0.0,
                           collision_penalty=-0.1)
        env.reset(seed=0)
        env.positions = [1, 3]
        res = env.step([1, 2])  # both move to cell 2 -> collision
        self.assertAlmostEqual(res.rewards[0], -0.1, places=5)
        self.assertAlmostEqual(res.rewards[1], -0.1, places=5)

    def test_finished_agent_freezes_and_stops_earning(self) -> None:
        # Once on its goal, an agent should not re-collect the goal reward.
        env = LineWorldEnv(n_agents=1, grid_length=5, goal_positions=[1],
                           step_penalty=-0.02, goal_reward=1.0, shared_goal_bonus=0.0,
                           max_steps=10)
        env.reset(seed=0)
        env.positions = [0]
        res1 = env.step([1])  # reach goal -> reward once
        self.assertAlmostEqual(res1.rewards[0], -0.02 + 1.0, places=5)
        res2 = env.step([1])  # frozen on goal -> no further reward, still at goal
        self.assertAlmostEqual(res2.rewards[0], 0.0, places=5)
        self.assertEqual(env.positions[0], 1)
        self.assertTrue(res2.terminated[0])

    def test_truncation_at_max_steps(self) -> None:
        env = LineWorldEnv(n_agents=1, grid_length=5, goal_positions=[4], max_steps=3)
        env.reset(seed=0)
        env.positions = [0]
        res = None
        for _ in range(3):
            res = env.step([0])  # stay
        self.assertTrue(res.truncated)

    def test_render_returns_rgb(self) -> None:
        env = LineWorldEnv(n_agents=2, grid_length=6)
        env.reset(seed=0)
        frame = env.render()
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)
        self.assertEqual(frame.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
