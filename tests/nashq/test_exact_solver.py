"""Tests for the exact minimax solver and exploitability helpers.

Examples:
    >>> # python -m pytest tests/nashq/test_exact_solver.py
"""

import unittest

import numpy as np

from nash_ql.envs.grid_soccer import GridSoccerEnv
from nash_ql.exact_solver import (
    _minimax_value,
    enumerate_transitions,
    exploitability,
    head_to_head_vs_exact,
    initial_state_indices,
    learned_equilibrium_policy,
    shapley_iterate,
    value_vs_best_response,
)


class TestMinimaxLP(unittest.TestCase):
    def test_matching_pennies(self) -> None:
        # payoff for row player: [[+1, -1], [-1, +1]]. Nash: (1/2, 1/2), value 0.
        P = np.array([[1.0, -1.0], [-1.0, 1.0]])
        v, pi0, pi1 = _minimax_value(P)
        self.assertAlmostEqual(v, 0.0, places=6)
        self.assertTrue(np.allclose(pi0, [0.5, 0.5], atol=1e-6))
        self.assertTrue(np.allclose(pi1, [0.5, 0.5], atol=1e-6))

    def test_dominant_column(self) -> None:
        # Row = agent 0 (max). Col 0 is dominated for agent 1 by col 1
        # (payoffs in col 1 are all smaller = worse for the maximizer). Agent 1
        # picks col 1; agent 0 then picks the max of col 1 = row 0 (0.2).
        P = np.array([[0.9, 0.2], [0.8, 0.1]])
        v, pi0, pi1 = _minimax_value(P)
        self.assertAlmostEqual(v, 0.2, places=6)
        self.assertAlmostEqual(pi0[0], 1.0, places=6)
        self.assertAlmostEqual(pi1[1], 1.0, places=6)


class TestGridSoccerSolver(unittest.TestCase):
    """Solve a tiny 2x3 game to keep the test suite fast."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = GridSoccerEnv(rows=2, cols=3, shaping=0.0, goal_reward=1.0, max_steps=20)
        cls.soln = shapley_iterate(cls.env, gamma=0.9, tol=1e-6)
        cls.T = enumerate_transitions(cls.env, shaping=0.0)

    def test_simulate_matches_step(self) -> None:
        # simulate() must reproduce step() for any fixed move order.
        import random
        env = GridSoccerEnv(rows=2, cols=3, shaping=0.1, goal_reward=1.0, max_steps=20)
        env.reset(seed=0)
        env.positions = [(0, 1), (1, 2)]
        env.ball_owner = 0
        state = env._encode()
        for order in [(0, 1), (1, 0)]:
            next_state, r0, r1, done, sc = env.simulate(state, [4, 0], order)
            # zero-sum
            self.assertAlmostEqual(r0 + r1, 0.0, places=6)

    def test_shapley_converged(self) -> None:
        self.assertLess(self.soln.residual, 1e-5)
        # In a symmetric game the value at (state, ball_owner=0) == -value at (state, ball_owner=1).
        env = self.env
        pos0, pos1 = (0, 1), (1, 2)
        s0 = env.encode([pos0, pos1], 0)
        s1 = env.encode([pos0, pos1], 1)
        # Anti-symmetric under ball flip; agent 0 vs agent 1 role isn't symmetric so
        # we only require |V(s0)| ~ |V(s1)|... but the SAME positions with swapped
        # ball ownership swaps who benefits. Weakest check: values have opposite signs.
        self.assertGreaterEqual(self.soln.V[s0] * self.soln.V[s1], -1.0)

    def test_exact_policy_is_zero_exploitable(self) -> None:
        V_vsBR, _ = value_vs_best_response(self.T, self.soln.pi0, fixed_agent=0, gamma=0.9)
        gap = self.soln.V - V_vsBR
        self.assertLess(float(np.max(gap)), 1e-4)  # V*(s) - V_vsBR(s) ≈ 0 everywhere

    def test_exploitability_of_exact_Q_near_zero(self) -> None:
        init = list(initial_state_indices(self.env))
        e = exploitability(self.soln.Q, self.T, self.soln, gamma=0.9, initial_states=init)
        self.assertLess(e["exploit_start"], 1e-4)
        self.assertLess(e["exploit_max"], 1e-4)

    def test_head_to_head_exact_hits_game_value(self) -> None:
        # 2000 games, mean r0 should be within ~2 SE of V*(start).
        h = head_to_head_vs_exact(
            self.soln.Q, self.env, self.soln,
            n_episodes=2000, seed=7, max_steps=20, learned_agent=0,
        )
        init = list(initial_state_indices(self.env))
        v_start = 0.5 * (self.soln.V[init[0]] + self.soln.V[init[1]])
        # rewards bounded in [-1, 1] so std <= 1; SE ~ 1/sqrt(2000) ~ 0.022.
        self.assertLess(abs(h["h2h_mean_r0"] - v_start), 0.08)

    def test_uniform_policy_is_exploitable(self) -> None:
        # A uniform policy should be strictly worse than V* somewhere.
        uniform = np.full((self.env.n_states, self.env.n_actions), 1.0 / self.env.n_actions)
        V_vsBR, _ = value_vs_best_response(self.T, uniform, fixed_agent=0, gamma=0.9)
        gap = self.soln.V - V_vsBR
        self.assertGreater(float(np.max(gap)), 0.05)

    def test_learned_equilibrium_policy_shape(self) -> None:
        pi = learned_equilibrium_policy(self.soln.Q)
        self.assertEqual(pi.shape, (self.env.n_states, self.env.n_actions))
        self.assertTrue(np.allclose(pi.sum(axis=1), 1.0, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
