from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class StepResult:
    observations: List[int]
    rewards: List[float]
    terminated: List[bool]
    truncated: bool
    info: Dict[str, Tuple[int, ...]]


class LineWorldEnv:
    """
    Simple cooperative line world for independent Q-learning.

    Agents begin on the left end of a one-dimensional grid and each has a
    personal goal cell toward the right end. Agents move simultaneously and are
    rewarded for reaching their own goal while avoiding collisions. When all
    reach their goals a shared bonus is provided to encourage coordination.
    """

    ACTION_MEANINGS = {
        0: "stay",
        1: "right",
        2: "left",
    }

    def __init__(
        self,
        n_agents: int = 2,
        grid_length: int = 7,
        max_steps: int = 60,
        goal_positions: Optional[Sequence[int]] = None,
        step_penalty: float = -0.02,
        goal_reward: float = 1.0,
        shared_goal_bonus: float = 0.5,
        collision_penalty: float = -0.1,
    ):
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1")
        if grid_length < 3:
            raise ValueError("grid_length must be >= 3")
        if goal_positions is not None and len(goal_positions) != n_agents:
            raise ValueError("goal_positions length must match n_agents")

        self.n_agents = n_agents
        self.grid_length = grid_length
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.shared_goal_bonus = shared_goal_bonus
        self.collision_penalty = collision_penalty

        if goal_positions is None:
            goal_positions = list(range(grid_length - n_agents, grid_length))
        for g in goal_positions:
            if not (0 <= g < grid_length):
                raise ValueError("goal positions must lie inside the grid")
        self.goal_positions = list(goal_positions)

        self._rng = random.Random()
        self._base = grid_length
        self._joint_state_space = self._base ** self.n_agents
        self.n_states = self._joint_state_space * self.n_agents
        self.n_actions = 3

        self.positions: List[int] = []
        self._steps = 0

    def reset(self, seed: Optional[int] = None) -> List[int]:
        if seed is not None:
            self._rng.seed(seed)
        self._steps = 0
        # Start agents evenly spaced from the left of the grid.
        self.positions = [min(i, self.grid_length - 1) for i in range(self.n_agents)]
        return self._encode_all()

    def step(self, actions: Sequence[int]) -> StepResult:
        if len(actions) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {len(actions)}")

        self._steps += 1
        proposed = list(self.positions)
        for idx, action in enumerate(actions):
            if action == 1 and proposed[idx] < self.grid_length - 1:
                proposed[idx] += 1
            elif action == 2 and proposed[idx] > 0:
                proposed[idx] -= 1

        rewards = [self.step_penalty for _ in range(self.n_agents)]
        if self.collision_penalty:
            counts: Dict[int, int] = {}
            for pos in proposed:
                counts[pos] = counts.get(pos, 0) + 1
            for idx, pos in enumerate(proposed):
                if counts[pos] > 1:
                    rewards[idx] += self.collision_penalty

        self.positions = proposed
        terminated = [False] * self.n_agents
        reached = 0
        for idx, pos in enumerate(self.positions):
            if pos == self.goal_positions[idx]:
                rewards[idx] += self.goal_reward
                terminated[idx] = True
                reached += 1

        if reached == self.n_agents and self.shared_goal_bonus:
            for idx in range(self.n_agents):
                rewards[idx] += self.shared_goal_bonus

        truncated = self._steps >= self.max_steps
        observations = self._encode_all()
        info = {"positions": tuple(self.positions)}
        return StepResult(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    # Convenience helpers -------------------------------------------------

    def action_meanings(self) -> Dict[int, str]:
        return dict(self.ACTION_MEANINGS)

    def _encode_all(self) -> List[int]:
        return [self._encode_state(agent_idx) for agent_idx in range(self.n_agents)]

    def _encode_state(self, agent_idx: int) -> int:
        code = 0
        for pos in self.positions:
            code = code * self._base + pos
        return agent_idx * self._joint_state_space + code
