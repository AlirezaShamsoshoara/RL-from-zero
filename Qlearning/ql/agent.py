from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    s: int
    a: int
    r: float
    s2: int
    done: bool  # True termination only, not an external time limit.


@dataclass
class QLearningStats:
    td_error: float
    target: float
    q_value: float


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
    ):
        for name, value in (("n_states", n_states), ("n_actions", n_actions)):
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be finite and in (0, 1]")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        if (
            not (np.isfinite(epsilon_start) and np.isfinite(epsilon_end))
            or not 0 <= epsilon_end <= epsilon_start <= 1
        ):
            raise ValueError("epsilon bounds must satisfy 0 <= end <= start <= 1")
        if not np.isfinite(epsilon_decay) or epsilon_decay < 0:
            raise ValueError("epsilon_decay must be finite and nonnegative")
        self.n_states, self.n_actions = int(n_states), int(n_actions)
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.eps_start, self.eps_end = float(epsilon_start), float(epsilon_end)
        self.eps_decay = float(epsilon_decay)
        self.global_step = 0
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    @staticmethod
    def _index(value, limit: int, name: str) -> int:
        if (
            not isinstance(value, (int, np.integer))
            or isinstance(value, bool)
            or not 0 <= value < limit
        ):
            raise ValueError(f"{name} index must be in [0, {limit})")
        return int(value)

    def epsilon(self) -> float:
        return float(
            self.eps_end
            + (self.eps_start - self.eps_end)
            * np.exp(-self.eps_decay * self.global_step)
        )

    def act(self, state: int, deterministic: bool = False) -> int:
        state = self._index(state, self.n_states, "state")
        if deterministic:
            # Evaluation neither consumes RNG nor advances the exploration schedule.
            return int(np.argmax(self.Q[state]))
        self.global_step += 1
        if np.random.rand() < self.epsilon():
            return int(np.random.randint(self.n_actions))
        candidates = np.flatnonzero(self.Q[state] == np.max(self.Q[state]))
        return int(np.random.choice(candidates))

    def update(self, tr: Transition) -> QLearningStats:
        state = self._index(tr.s, self.n_states, "state")
        action = self._index(tr.a, self.n_actions, "action")
        next_state = self._index(tr.s2, self.n_states, "next state")
        if not np.isfinite(tr.r):
            raise ValueError("reward must be finite")
        target = float(tr.r) + (
            0.0 if tr.done else self.gamma * float(np.max(self.Q[next_state]))
        )
        td_error = target - float(self.Q[state, action])
        updated = float(self.Q[state, action]) + self.alpha * td_error
        if not np.all(np.isfinite([target, td_error, updated])) or abs(updated) > float(
            np.finfo(np.float32).max
        ):
            raise ValueError("TD update exceeds the finite Q-table range")
        self.Q[state, action] = updated
        return QLearningStats(
            td_error=td_error, target=target, q_value=float(self.Q[state, action])
        )

    def state_dict(self) -> dict:
        return {"q_table": self.Q.copy(), "global_step": self.global_step}

    def load_state_dict(self, state: dict) -> None:
        table = state["q_table"]
        if hasattr(table, "detach"):
            table = table.detach().cpu().numpy()
        table = np.asarray(table, dtype=np.float32)
        if table.shape != (self.n_states, self.n_actions) or not np.all(
            np.isfinite(table)
        ):
            raise ValueError(
                "checkpoint Q-table has an invalid shape or non-finite values"
            )
        step = state.get("global_step", 0)
        if (
            not isinstance(step, (int, np.integer))
            or isinstance(step, bool)
            or step < 0
        ):
            raise ValueError("global_step must be a nonnegative integer")
        self.Q = table.copy()
        self.global_step = int(step)
