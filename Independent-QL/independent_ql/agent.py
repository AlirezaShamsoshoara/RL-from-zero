from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


@dataclass
class Transition:
    agent: int
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class IndependentQLearningAgent:
    def __init__(
        self,
        n_agents: int,
        n_states: int,
        n_actions: int,
        alpha: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
    ):
        self.n_agents = n_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay
        self.global_step = 0
        self.Q = np.zeros((n_agents, n_states, n_actions), dtype=np.float32)

    def epsilon(self) -> float:
        eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -self.eps_decay * max(self.global_step - 1, 0)
        )
        return float(np.clip(eps, self.eps_end, self.eps_start))

    def act(self, states: Sequence[int]) -> List[int]:
        if len(states) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} states, got {len(states)}")
        self.global_step += 1
        eps = self.epsilon()
        actions: List[int] = []
        for agent_idx, state in enumerate(states):
            if state >= self.n_states or state < 0:
                raise ValueError(f"State index {state} out of bounds for agent {agent_idx}")
            if np.random.rand() < eps:
                action = int(np.random.randint(self.n_actions))
            else:
                q_vals = self.Q[agent_idx, state]
                max_q = np.max(q_vals)
                candidates = np.flatnonzero(q_vals == max_q)
                action = int(np.random.choice(candidates))
            actions.append(action)
        return actions

    def greedy_actions(self, states: Sequence[int]) -> List[int]:
        if len(states) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} states, got {len(states)}")
        actions: List[int] = []
        for agent_idx, state in enumerate(states):
            q_vals = self.Q[agent_idx, state]
            actions.append(int(np.argmax(q_vals)))
        return actions

    def update(self, transitions: Iterable[Transition]) -> None:
        for tr in transitions:
            target = tr.reward
            if not tr.done:
                target += self.gamma * float(np.max(self.Q[tr.agent, tr.next_state]))
            td_error = target - self.Q[tr.agent, tr.state, tr.action]
            self.Q[tr.agent, tr.state, tr.action] += self.alpha * td_error
