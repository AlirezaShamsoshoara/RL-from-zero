from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import linprog


def solve_nash_equilibrium(
    payoff_1: np.ndarray, payoff_2: np.ndarray, tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for Nash equilibrium of a 2-player general-sum game using linear programming.

    Args:
        payoff_1: Payoff matrix for agent 1, shape (n_actions_1, n_actions_2)
        payoff_2: Payoff matrix for agent 2, shape (n_actions_1, n_actions_2)
        tol: Numerical tolerance for probability clipping

    Returns:
        (pi_1, pi_2): Nash equilibrium mixed strategies
            pi_1: Probability distribution over agent 1's actions, shape (n_actions_1,)
            pi_2: Probability distribution over agent 2's actions, shape (n_actions_2,)

    Note:
        For 2-player general-sum games, we solve for a Nash equilibrium using the
        support enumeration method implemented via linear programming. This finds
        one Nash equilibrium (games may have multiple).
    """
    n_actions_1, n_actions_2 = payoff_1.shape

    if payoff_2.shape != (n_actions_1, n_actions_2):
        raise ValueError("Payoff matrices must have the same shape")

    # Special case: if one player has only one action, return deterministic strategy
    if n_actions_1 == 1:
        return np.array([1.0]), solve_single_player(payoff_2, axis=0)
    if n_actions_2 == 1:
        return solve_single_player(payoff_1, axis=1), np.array([1.0])

    # Try to solve for agent 1's strategy using LP
    # We want: max v1 s.t. Q1^T @ pi2 >= v1 * 1, sum(pi2) = 1, pi2 >= 0
    # Reformulate as: min -v1 s.t. -Q1^T @ pi2 + v1 * 1 <= 0, sum(pi2) = 1, pi2 >= 0
    #
    # Variables: [v1, pi2_0, pi2_1, ..., pi2_{n2-1}]
    # Objective: minimize -v1 (i.e., maximize v1)
    c = np.zeros(1 + n_actions_2)
    c[0] = -1.0  # Maximize v1

    # Inequality constraints: -Q1^T @ pi2 + v1 * 1 <= 0
    # For each action a1 of agent 1: -sum_a2 Q1[a1, a2] * pi2[a2] + v1 <= 0
    A_ub = np.zeros((n_actions_1, 1 + n_actions_2))
    A_ub[:, 0] = 1.0  # v1 coefficient
    A_ub[:, 1:] = -payoff_1  # -Q1
    b_ub = np.zeros(n_actions_1)

    # Equality constraint: sum(pi2) = 1
    A_eq = np.zeros((1, 1 + n_actions_2))
    A_eq[0, 1:] = 1.0
    b_eq = np.array([1.0])

    # Bounds: v1 unbounded, pi2 >= 0
    bounds = [(None, None)] + [(0, 1) for _ in range(n_actions_2)]

    # Solve LP for agent 2's strategy
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    if not result.success:
        # Fallback: uniform distribution
        pi_2 = np.ones(n_actions_2) / n_actions_2
    else:
        pi_2 = result.x[1:]
        pi_2 = np.clip(pi_2, 0, 1)
        pi_2 = pi_2 / (pi_2.sum() + 1e-10)

    # Now solve for agent 1's strategy given agent 2's (best response)
    # We want: max v2 s.t. Q2 @ pi1 >= v2 * 1, sum(pi1) = 1, pi1 >= 0
    c = np.zeros(1 + n_actions_1)
    c[0] = -1.0  # Maximize v2

    # Inequality constraints: -Q2 @ pi1 + v2 * 1 <= 0
    A_ub = np.zeros((n_actions_2, 1 + n_actions_1))
    A_ub[:, 0] = 1.0  # v2 coefficient
    A_ub[:, 1:] = -payoff_2.T  # -Q2^T
    b_ub = np.zeros(n_actions_2)

    # Equality constraint: sum(pi1) = 1
    A_eq = np.zeros((1, 1 + n_actions_1))
    A_eq[0, 1:] = 1.0
    b_eq = np.array([1.0])

    # Bounds: v2 unbounded, pi1 >= 0
    bounds = [(None, None)] + [(0, 1) for _ in range(n_actions_1)]

    # Solve LP for agent 1's strategy
    result = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )

    if not result.success:
        # Fallback: uniform distribution
        pi_1 = np.ones(n_actions_1) / n_actions_1
    else:
        pi_1 = result.x[1:]
        pi_1 = np.clip(pi_1, 0, 1)
        pi_1 = pi_1 / (pi_1.sum() + 1e-10)

    # Clean up probabilities
    pi_1[pi_1 < tol] = 0.0
    pi_1 = pi_1 / (pi_1.sum() + 1e-10)

    pi_2[pi_2 < tol] = 0.0
    pi_2 = pi_2 / (pi_2.sum() + 1e-10)

    return pi_1, pi_2


def solve_single_player(payoff: np.ndarray, axis: int) -> np.ndarray:
    """
    Solve single-player decision problem (best response).

    Args:
        payoff: Payoff matrix
        axis: Axis along which to maximize (0 for rows, 1 for columns)

    Returns:
        Probability distribution over actions (uniform over best actions)
    """
    if axis == 0:
        # Maximize over rows (agent 2's strategy when agent 1 has single action)
        q_vals = payoff[0, :]
    else:
        # Maximize over columns (agent 1's strategy when agent 2 has single action)
        q_vals = payoff[:, 0]

    max_q = np.max(q_vals)
    best_actions = np.where(np.abs(q_vals - max_q) < 1e-8)[0]

    pi = np.zeros(len(q_vals))
    pi[best_actions] = 1.0 / len(best_actions)

    return pi


def compute_nash_value(
    payoff_1: np.ndarray, payoff_2: np.ndarray, pi_1: np.ndarray, pi_2: np.ndarray
) -> Tuple[float, float]:
    """
    Compute expected payoffs for both agents under given mixed strategies.

    Args:
        payoff_1: Payoff matrix for agent 1, shape (n_actions_1, n_actions_2)
        payoff_2: Payoff matrix for agent 2, shape (n_actions_1, n_actions_2)
        pi_1: Mixed strategy for agent 1, shape (n_actions_1,)
        pi_2: Mixed strategy for agent 2, shape (n_actions_2,)

    Returns:
        (v1, v2): Expected payoffs for agent 1 and agent 2
    """
    v1 = float(pi_1 @ payoff_1 @ pi_2)
    v2 = float(pi_1 @ payoff_2 @ pi_2)
    return v1, v2
