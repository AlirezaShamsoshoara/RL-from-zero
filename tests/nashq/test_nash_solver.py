"""Nash equilibrium solver tests.

Examples:
    >>> # python -m pytest tests/nashq/test_nash_solver.py
"""

import unittest

import numpy as np

from nash_ql.nash_solver import (
    compute_nash_value,
    solve_nash_equilibrium,
    solve_single_player,
)


class TestNashSolver(unittest.TestCase):
    def test_returns_valid_distributions(self) -> None:
        rng = np.random.default_rng(0)
        p1 = rng.standard_normal((3, 3)).astype(np.float32)
        p2 = rng.standard_normal((3, 3)).astype(np.float32)
        pi1, pi2 = solve_nash_equilibrium(p1, p2)
        self.assertEqual(pi1.shape, (3,))
        self.assertEqual(pi2.shape, (3,))
        for pi in (pi1, pi2):
            self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)
            self.assertTrue((pi >= -1e-8).all())

    def test_mismatched_shapes_raise(self) -> None:
        with self.assertRaises(ValueError):
            solve_nash_equilibrium(np.zeros((2, 3)), np.zeros((3, 2)))

    def test_single_action_player(self) -> None:
        # Agent 1 has a single action; agent 2 picks its best column.
        p1 = np.array([[0.0, 0.0, 0.0]])
        p2 = np.array([[1.0, 5.0, 2.0]])
        pi1, pi2 = solve_nash_equilibrium(p1, p2)
        self.assertEqual(pi1.shape, (1,))
        self.assertAlmostEqual(float(pi1[0]), 1.0, places=6)
        self.assertEqual(int(np.argmax(pi2)), 1)  # column 1 has the highest payoff

    def test_solve_single_player_uniform_over_best(self) -> None:
        payoff = np.array([[1.0, 3.0, 3.0]])  # axis 0: best columns are 1 and 2
        pi = solve_single_player(payoff, axis=0)
        self.assertAlmostEqual(float(pi[1]), 0.5, places=6)
        self.assertAlmostEqual(float(pi[2]), 0.5, places=6)
        self.assertAlmostEqual(float(pi[0]), 0.0, places=6)

    def test_compute_nash_value_matches_quadratic_form(self) -> None:
        p1 = np.array([[1.0, 0.0], [0.0, 2.0]])
        p2 = np.array([[2.0, 0.0], [0.0, 1.0]])
        pi1 = np.array([0.5, 0.5])
        pi2 = np.array([0.5, 0.5])
        v1, v2 = compute_nash_value(p1, p2, pi1, pi2)
        self.assertAlmostEqual(v1, float(pi1 @ p1 @ pi2), places=6)
        self.assertAlmostEqual(v2, float(pi1 @ p2 @ pi2), places=6)

    def test_matching_pennies_is_uniform(self) -> None:
        # The solver computes maxmin (security) strategies via LP, which is the
        # exact Nash equilibrium for a zero-sum game. Matching pennies has the
        # unique equilibrium (0.5, 0.5) for both players.
        p1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
        p2 = -p1
        pi1, pi2 = solve_nash_equilibrium(p1, p2)
        self.assertAlmostEqual(float(pi1[0]), 0.5, places=2)
        self.assertAlmostEqual(float(pi2[0]), 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
