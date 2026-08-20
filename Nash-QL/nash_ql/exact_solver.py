"""Exact minimax value iteration (Shapley 1953) for a two-player zero-sum
stochastic game, plus best-response value iteration for exploitability.

Applied to :mod:`nash_ql.envs.grid_soccer` this gives:
  * ``Q*[s, a0, a1]`` and ``V*[s]`` for the true, unshaped +/-1 game;
  * per-state analytical Nash policies ``pi0*[s], pi1*[s]``;
  * ``value_vs_best_response(policy, opponent)`` for any learned policy, so
    exploitability := V*(s0) - value_vs_BR(learned_policy) can be measured.

The dynamics come from the env's pure ``simulate(state, actions, order)`` helper.
Stochasticity comes only from the random move order (steal mechanic), so each
(state, joint action) has exactly two equally-likely outcomes; the solver
enumerates both.

Runtime on default 4x5 soccer (800 states, 25 joint actions): a few minutes for
Shapley VI to converge; a few seconds for best-response VI. Results are cached
to disk keyed by env config so training does not re-solve every run.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog


@dataclass
class Transitions:
    """Vectorized transition model. Each (s, a0, a1) has exactly 2 outcomes
    corresponding to the two possible move orders (equally likely). Terminal
    states self-loop with zero reward. Shape ``(n_states, n_actions, n_actions, 2)``.
    """

    next_state: np.ndarray   # int64, shape (n_s, n_a, n_a, 2)
    reward: np.ndarray       # float64, agent 0's reward
    done: np.ndarray         # bool


@dataclass
class ExactSolution:
    """Analytical solution of the (unshaped, zero-sum) stochastic game."""

    Q: np.ndarray            # shape (n_states, n_actions, n_actions), agent 0 payoff
    V: np.ndarray            # shape (n_states,)
    pi0: np.ndarray          # shape (n_states, n_actions) - Nash policy for agent 0
    pi1: np.ndarray          # shape (n_states, n_actions) - Nash policy for agent 1
    gamma: float
    iters: int
    residual: float


def _terminal_states(env) -> np.ndarray:
    term = np.zeros(env.n_states, dtype=bool)
    for s in range(env.n_states):
        pos0, pos1, bo = env.decode(s)
        if bo == 0 and env._is_goal_cell(0, pos0):
            term[s] = True
        elif bo == 1 and env._is_goal_cell(1, pos1):
            term[s] = True
    return term


def enumerate_transitions(env, shaping: float = 0.0) -> Transitions:
    """Precompute vectorized (next_state, r0, done) tensors for every (s, a0, a1, order)."""
    n_s, n_a = env.n_states, env.n_actions
    orders = [(0, 1), (1, 0)]
    ns = np.zeros((n_s, n_a, n_a, 2), dtype=np.int64)
    r0 = np.zeros((n_s, n_a, n_a, 2), dtype=np.float64)
    done = np.zeros((n_s, n_a, n_a, 2), dtype=bool)
    term = _terminal_states(env)
    for s in range(n_s):
        if term[s]:
            ns[s, :, :, :] = s          # absorbing self-loop
            done[s, :, :, :] = True
            continue
        for a0 in range(n_a):
            for a1 in range(n_a):
                for oi, order in enumerate(orders):
                    nxt, rw, _r1, dn, _sc = env.simulate(s, [a0, a1], order, shaping=shaping)
                    ns[s, a0, a1, oi] = nxt
                    r0[s, a0, a1, oi] = rw
                    done[s, a0, a1, oi] = dn
    return Transitions(next_state=ns, reward=r0, done=done)


def _bellman_backup_zero_sum(T: Transitions, V: np.ndarray, gamma: float) -> np.ndarray:
    """Q[s, a0, a1] = 0.5 * sum_{o in {0,1}} (r0[s,a,o] + gamma * V[s'[s,a,o]] * (1 - done))."""
    V_next = V[T.next_state]                       # (n_s, n_a, n_a, 2)
    future = np.where(T.done, 0.0, gamma * V_next)
    per_order = T.reward + future                  # (n_s, n_a, n_a, 2)
    return 0.5 * per_order.sum(axis=-1)            # (n_s, n_a, n_a)


def _minimax_value(payoff: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Exact minimax of a zero-sum stage game (agent 0 maximizes ``payoff``).

    Solves the standard pair of dual LPs. By the minimax theorem both give the
    same value V* = max_{pi0} min_{a1} pi0.T P e_{a1} = min_{pi1} max_{a0} ...;
    ``solve_nash_equilibrium`` in nash_solver.py is a general-sum heuristic and
    silently returns wrong values on many zero-sum matrices, so we do this
    directly here.
    """
    n0, n1 = payoff.shape
    # Agent 0's max strategy: max V s.t. payoff.T @ pi0 >= V * 1, sum(pi0)=1, pi0>=0.
    # Vars = [V, pi0_0, ..., pi0_{n0-1}]. Minimize -V.
    c = np.zeros(1 + n0); c[0] = -1.0
    A_ub = np.zeros((n1, 1 + n0))
    A_ub[:, 0] = 1.0
    A_ub[:, 1:] = -payoff.T
    b_ub = np.zeros(n1)
    A_eq = np.zeros((1, 1 + n0)); A_eq[0, 1:] = 1.0
    b_eq = np.ones(1)
    bounds = [(None, None)] + [(0.0, 1.0)] * n0
    r0 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not r0.success:
        raise RuntimeError(f"maxmin LP failed: {r0.message}")
    v = float(-r0.fun)
    pi0 = np.clip(r0.x[1:], 0.0, 1.0)
    pi0 = pi0 / pi0.sum()
    # Agent 1's min strategy: min U s.t. payoff @ pi1 <= U * 1, sum(pi1)=1, pi1>=0.
    c = np.zeros(1 + n1); c[0] = 1.0
    A_ub = np.zeros((n0, 1 + n1))
    A_ub[:, 0] = -1.0
    A_ub[:, 1:] = payoff
    b_ub = np.zeros(n0)
    A_eq = np.zeros((1, 1 + n1)); A_eq[0, 1:] = 1.0
    b_eq = np.ones(1)
    bounds = [(None, None)] + [(0.0, 1.0)] * n1
    r1 = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not r1.success:
        raise RuntimeError(f"minmax LP failed: {r1.message}")
    pi1 = np.clip(r1.x[1:], 0.0, 1.0)
    pi1 = pi1 / pi1.sum()
    return v, pi0, pi1


def shapley_iterate(
    env,
    gamma: float,
    shaping: float = 0.0,
    tol: float = 1e-6,
    max_iters: int = 400,
    verbose: bool = False,
) -> ExactSolution:
    """Infinite-horizon zero-sum Nash-VI (Shapley 1953). Contraction with rate gamma."""
    T = enumerate_transitions(env, shaping=shaping)
    n_s, n_a = env.n_states, env.n_actions
    V = np.zeros(n_s, dtype=np.float64)
    pi0 = np.full((n_s, n_a), 1.0 / n_a)
    pi1 = np.full((n_s, n_a), 1.0 / n_a)
    Q = np.zeros((n_s, n_a, n_a), dtype=np.float64)
    residual = float("inf")
    t0 = time.time()
    for it in range(1, max_iters + 1):
        Q = _bellman_backup_zero_sum(T, V, gamma)
        V_new = np.zeros_like(V)
        for s in range(n_s):
            v, p0, p1 = _minimax_value(Q[s])
            V_new[s] = v
            pi0[s] = p0
            pi1[s] = p1
        residual = float(np.max(np.abs(V_new - V)))
        V = V_new
        if verbose and (it % 10 == 0 or residual < tol):
            print(f"  iter {it:3d}  residual={residual:.3e}  ({time.time()-t0:.1f}s)")
        if residual < tol:
            break
    return ExactSolution(Q=Q, V=V, pi0=pi0, pi1=pi1, gamma=gamma, iters=it, residual=residual)


# ---- best-response value iteration (for exploitability) --------------------

def value_vs_best_response(
    T: Transitions,
    policy: np.ndarray,       # shape (n_states, n_actions)
    fixed_agent: int,         # 0 or 1: which agent is playing `policy`
    gamma: float,
    tol: float = 1e-7,
    max_iters: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Value function when `fixed_agent` plays `policy` and the other plays best-response.

    Returns ``(V, br_policy)`` where V is the value achieved BY agent 0 (the
    zero-sum payoff owner) at every state under (policy, best_response).

    If fixed_agent=0, the opponent (agent 1) minimizes r0 => V(s) = min_{a1}
    E_{a0 ~ pi(s)}[Q0(s, a0, a1)]. If fixed_agent=1, agent 0 maximizes r0.
    """
    n_s = T.next_state.shape[0]
    V = np.zeros(n_s, dtype=np.float64)
    br = np.zeros(n_s, dtype=np.int64)
    for _ in range(max_iters):
        Q = _bellman_backup_zero_sum(T, V, gamma)  # (n_s, n_a, n_a); agent 0 payoff
        if fixed_agent == 0:
            # Fix agent 0 to `policy`; opponent (agent 1) minimizes over a1.
            q_marg = np.einsum("sa,sab->sb", policy, Q)  # (n_s, n_a)
            br = np.argmin(q_marg, axis=1)
            V_new = q_marg[np.arange(n_s), br]
        else:
            # Fix agent 1 to `policy`; agent 0 maximizes over a0.
            q_marg = np.einsum("sab,sb->sa", Q, policy)
            br = np.argmax(q_marg, axis=1)
            V_new = q_marg[np.arange(n_s), br]
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return V, br


# ---- policy extraction + exploitability + head-to-head --------------------

def learned_equilibrium_policy(Q0: np.ndarray) -> np.ndarray:
    """Extract agent 0's stochastic Nash policy from a learned Q table.

    Solves the stage-game minimax LP at every state of ``Q0`` (shape
    ``(n_states, n_actions, n_actions)``). Returns ``pi[s]`` = agent 0's mixed
    equilibrium strategy of the learned payoff matrix at s.
    """
    n_s, n_a, _ = Q0.shape
    pi = np.zeros((n_s, n_a), dtype=np.float64)
    for s in range(n_s):
        _, p0, _ = _minimax_value(Q0[s])
        pi[s] = p0
    return pi


def initial_state_indices(env) -> Tuple[int, int]:
    """Encode the two possible starting states (agent 0 or agent 1 holds the ball)."""
    mid = env.rows // 2
    p0 = (mid, 1)
    p1 = (mid, env.cols - 2)
    return env.encode([p0, p1], 0), env.encode([p0, p1], 1)


def exploitability(
    Q0_learned: np.ndarray,
    T_true: Transitions,
    exact: ExactSolution,
    gamma: float,
    initial_states: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """How much can a best-responding opponent exploit the learned equilibrium.

    Extracts agent 0's learned equilibrium policy from ``Q0_learned``, then runs
    best-response VI on the true (unshaped) game to measure how far the learned
    policy is from V* in value terms. Returns per-initial-state exploitability
    and a state-averaged summary. Zero iff the learned policy is a genuine Nash
    equilibrium of the true game.
    """
    pi = learned_equilibrium_policy(Q0_learned)
    V_vsBR, _ = value_vs_best_response(T_true, pi, fixed_agent=0, gamma=gamma)
    gap = exact.V - V_vsBR  # >= 0 in theory (opponent BR can only lower value)
    out: Dict[str, float] = {"exploit_mean": float(gap.mean()),
                             "exploit_max": float(gap.max())}
    if initial_states:
        out["exploit_start"] = float(np.mean([gap[s] for s in initial_states]))
    return out


def head_to_head_vs_exact(
    Q0_learned: np.ndarray,
    env,
    exact: ExactSolution,
    n_episodes: int,
    seed: int,
    max_steps: int,
    learned_agent: int = 0,
) -> Dict[str, float]:
    """Simulate the learned equilibrium policy against the analytical Nash opponent.

    Runs unshaped games (goals only) so returns are directly comparable to V*.
    Reports win / draw / loss rates for the learned agent and the mean undiscounted
    agent-0 return; the latter should approach V*(s0) as Nash-Q converges.
    """
    import random as _random
    rng = _random.Random(seed)

    pi_learned = learned_equilibrium_policy(Q0_learned)
    pi_exact_opp = exact.pi1 if learned_agent == 0 else exact.pi0

    wins = losses = draws = 0
    total_r0 = 0.0
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        state = env.encode(env.positions, env.ball_owner)
        ep_r0 = 0.0
        scorer = None
        for step in range(max_steps):
            if learned_agent == 0:
                a_learn = int(rng.choices(range(env.n_actions), weights=pi_learned[state])[0])
                a_opp = int(rng.choices(range(env.n_actions), weights=pi_exact_opp[state])[0])
                actions = [a_learn, a_opp]
            else:
                a_opp = int(rng.choices(range(env.n_actions), weights=pi_exact_opp[state])[0])
                a_learn = int(rng.choices(range(env.n_actions), weights=pi_learned[state])[0])
                actions = [a_opp, a_learn]
            order = (0, 1) if rng.random() < 0.5 else (1, 0)
            next_state, r0, _r1, done, sc = env.simulate(state, actions, order, shaping=0.0)
            ep_r0 += r0
            state = next_state
            if done:
                scorer = sc
                break
        total_r0 += ep_r0
        if scorer == learned_agent:
            wins += 1
        elif scorer is None:
            draws += 1
        else:
            losses += 1
    n = float(n_episodes)
    return {
        "h2h_win": wins / n,
        "h2h_draw": draws / n,
        "h2h_loss": losses / n,
        "h2h_mean_r0": total_r0 / n,
    }


# ---- disk cache ------------------------------------------------------------

def _cache_key(env_kwargs: Dict, gamma: float, shaping: float, tol: float) -> str:
    payload = json.dumps({"env_kwargs": env_kwargs, "gamma": gamma,
                          "shaping": shaping, "tol": tol}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def load_or_solve(
    env,
    env_kwargs: Dict,
    gamma: float,
    cache_dir: str,
    shaping: float = 0.0,
    tol: float = 1e-6,
    force: bool = False,
    verbose: bool = False,
) -> ExactSolution:
    """Load a cached exact solution or compute + cache one."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(env_kwargs, gamma, shaping, tol)
    path = os.path.join(cache_dir, f"exact_soln_{key}.npz")
    if not force and os.path.exists(path):
        data = np.load(path)
        return ExactSolution(
            Q=data["Q"], V=data["V"], pi0=data["pi0"], pi1=data["pi1"],
            gamma=float(data["gamma"]), iters=int(data["iters"]),
            residual=float(data["residual"]),
        )
    if verbose:
        print(f"[exact_solver] no cache; solving (key={key})...")
    soln = shapley_iterate(env, gamma=gamma, shaping=shaping, tol=tol, verbose=verbose)
    np.savez(path, Q=soln.Q, V=soln.V, pi0=soln.pi0, pi1=soln.pi1,
             gamma=soln.gamma, iters=soln.iters, residual=soln.residual)
    if verbose:
        print(f"[exact_solver] cached to {path}")
    return soln
