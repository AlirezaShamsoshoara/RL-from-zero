"""Grid soccer environment tests.

Examples:
    >>> # python -m pytest tests/nashq/test_grid_soccer.py
"""

import unittest

import numpy as np

from nash_ql.envs.grid_soccer import GridSoccerEnv


class TestGridSoccer(unittest.TestCase):
    def test_shapes_and_shared_state(self) -> None:
        env = GridSoccerEnv(rows=4, cols=5)
        states = env.reset(seed=0)
        self.assertEqual(env.n_agents, 2)
        self.assertEqual(env.n_actions, 5)
        self.assertEqual(env.n_states, (4 * 5) ** 2 * 2)
        self.assertEqual(states[0], states[1])  # shared joint state index
        self.assertTrue(0 <= states[0] < env.n_states)

    def test_scoring_zero_sum(self) -> None:
        env = GridSoccerEnv(rows=4, cols=5, goal_reward=1.0, shaping=0.0)
        env.reset(seed=0)
        # Put agent 0 next to the right goal (middle row) with the ball.
        env.positions = [(1, 3), (3, 0)]
        env.ball_owner = 0
        res = env.step([4, 0])  # agent 0 moves right into (1, 4) = its goal
        self.assertEqual(res.info["scorer"], 0)
        self.assertEqual(res.rewards, [1.0, -1.0])  # zero-sum
        self.assertTrue(all(res.terminated))

    def test_shaping_is_zero_sum_and_rewards_progress(self) -> None:
        env = GridSoccerEnv(rows=4, cols=5, goal_reward=1.0, shaping=0.1)
        env.reset(seed=0)
        env.positions = [(0, 1), (3, 4)]  # agent 0 has ball, away from right goal
        env.ball_owner = 0
        res = env.step([4, 0])  # agent 0 advances right (closer to its goal)
        self.assertAlmostEqual(res.rewards[0] + res.rewards[1], 0.0, places=6)
        self.assertGreater(res.rewards[0], 0.0)  # carrier rewarded for progress

    def test_steal_on_collision(self) -> None:
        env = GridSoccerEnv(rows=4, cols=5)
        env.reset(seed=0)
        env.positions = [(1, 2), (1, 3)]
        env.ball_owner = 0
        env.step([4, 0])  # agent 0 (ball) moves into agent 1's cell -> steal
        self.assertEqual(env.ball_owner, 1)
        self.assertEqual(env.positions[0], (1, 2))  # blocked, stayed

    def test_no_score_without_ball(self) -> None:
        env = GridSoccerEnv(rows=4, cols=5, shaping=0.0)
        env.reset(seed=0)
        env.positions = [(1, 3), (3, 0)]
        env.ball_owner = 1  # agent 0 does NOT have the ball
        res = env.step([4, 0])  # agent 0 reaches its goal cell but has no ball
        self.assertIsNone(res.info["scorer"])
        self.assertEqual(res.rewards, [0.0, 0.0])

    def test_truncation(self) -> None:
        env = GridSoccerEnv(rows=4, cols=5, max_steps=3)
        env.reset(seed=0)
        res = None
        for _ in range(3):
            res = env.step([0, 0])  # both stand -> no goal
        self.assertTrue(res.truncated)

    def test_render_rgb(self) -> None:
        env = GridSoccerEnv(rows=4, cols=5)
        env.reset(seed=0)
        frame = env.render()
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)
        self.assertEqual(frame.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
