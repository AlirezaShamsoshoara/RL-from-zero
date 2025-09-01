from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Transition:
    s: int
    a: int
    r: float
    s2: int
    done: bool


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
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay
        self.global_step = 0
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def epsilon(self) -> float:
        # Exponential decay of epsilon per step
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -self.eps_decay * self.global_step
        )
        return float(eps)

    def act(self, state: int) -> int:
        self.global_step += 1
        if np.random.rand() < self.epsilon():
            return int(np.random.randint(self.n_actions))
        # Greedy action with random tie-break
        q_vals = self.Q[state]
        max_q = np.max(q_vals)
        candidates = np.flatnonzero(q_vals == max_q)
        return int(np.random.choice(candidates))

    def update(self, tr: Transition):
        target = tr.r + (0.0 if tr.done else self.gamma * np.max(self.Q[tr.s2]))
        td_error = target - self.Q[tr.s, tr.a]
        self.Q[tr.s, tr.a] += self.alpha * td_error

