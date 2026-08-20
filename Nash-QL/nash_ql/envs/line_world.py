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
    Cooperative line world for Nash Q-learning.

    Agents begin on the left of a one-dimensional grid and each has a personal
    goal cell toward the right. Agents move simultaneously and are rewarded for
    reaching their own goal while avoiding collisions; a shared bonus is given
    when all agents are on their goals.

    State encoding: every agent observes the SAME joint state index (the tuple of
    all positions, base-`grid_length`). Nash Q-learning assumes a fully observable
    shared state at which both agents' stage-game payoff matrices are indexed, so
    the observation is intentionally identical across agents.
    """

    ACTION_MEANINGS = {
        0: "stay",
        1: "right",
        2: "left",
    }

    _PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed"]

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
        # Shared joint state: all agents see the same index, so Nash Q-tables
        # Q[agent, state, a1, a2] are indexed consistently for both agents.
        self.n_states = self._base ** self.n_agents
        self.n_actions = 3

        self.positions: List[int] = []
        self._steps = 0
        self._done: List[bool] = []
        self._bonus_given = False

    def reset(self, seed: Optional[int] = None) -> List[int]:
        if seed is not None:
            self._rng.seed(seed)
        self._steps = 0
        self.positions = [min(i, self.grid_length - 1) for i in range(self.n_agents)]
        # Once an agent reaches its goal it freezes there (no move, no further
        # reward), so a finished agent cannot farm the goal reward while waiting.
        self._done = [False] * self.n_agents
        self._bonus_given = False
        return self._encode_all()

    def step(self, actions: Sequence[int]) -> StepResult:
        if len(actions) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {len(actions)}")

        self._steps += 1
        proposed = list(self.positions)
        for idx, action in enumerate(actions):
            if self._done[idx]:
                continue  # finished agents are frozen on their goal
            if action == 1 and proposed[idx] < self.grid_length - 1:
                proposed[idx] += 1
            elif action == 2 and proposed[idx] > 0:
                proposed[idx] -= 1

        rewards = [0.0 if self._done[idx] else self.step_penalty for idx in range(self.n_agents)]
        if self.collision_penalty:
            counts: Dict[int, int] = {}
            for pos in proposed:
                counts[pos] = counts.get(pos, 0) + 1
            for idx, pos in enumerate(proposed):
                if not self._done[idx] and counts[pos] > 1:
                    rewards[idx] += self.collision_penalty

        self.positions = proposed
        for idx, pos in enumerate(self.positions):
            if not self._done[idx] and pos == self.goal_positions[idx]:
                rewards[idx] += self.goal_reward  # granted once, on arrival
                self._done[idx] = True

        if all(self._done) and self.shared_goal_bonus and not self._bonus_given:
            for idx in range(self.n_agents):
                rewards[idx] += self.shared_goal_bonus
            self._bonus_given = True

        terminated = list(self._done)
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
        code = self._encode_joint()
        return [code for _ in range(self.n_agents)]

    def _encode_joint(self) -> int:
        code = 0
        for pos in self.positions:
            code = code * self._base + pos
        return code

    def render(self, title: Optional[str] = None):
        """Render the current state to an RGB numpy array (H, W, 3).

        Draws the 1D grid as a row of cells; each agent is a filled circle and
        its goal is a hollow star in the same color, one stacked lane per agent.
        Used by the demo GIF tool.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        palette = self._PALETTE
        fig, ax = plt.subplots(figsize=(max(4, self.grid_length), 1.2 + 0.7 * self.n_agents), dpi=80)
        for x in range(self.grid_length):
            ax.add_patch(plt.Rectangle((x - 0.5, -0.5), 1.0, self.n_agents,
                                       fill=False, edgecolor="#cccccc", lw=1.0))
            ax.text(x, -0.85, str(x), ha="center", va="center", fontsize=8, color="#999999")
        for idx in range(self.n_agents):
            color = palette[idx % len(palette)]
            lane = self.n_agents - 1 - idx
            ax.scatter([self.goal_positions[idx]], [lane], marker="*", s=420,
                       facecolors="none", edgecolors=color, linewidths=2.0, zorder=2)
            at_goal = self.positions[idx] == self.goal_positions[idx]
            ax.scatter([self.positions[idx]], [lane], s=320, color=color,
                       edgecolors="black" if at_goal else color, linewidths=2.0, zorder=3)
            ax.text(-1.2, lane, f"agent {idx}", ha="right", va="center", fontsize=9, color=color)
        ax.set_xlim(-1.6, self.grid_length - 0.4)
        ax.set_ylim(-1.2, self.n_agents - 0.2)
        ax.set_title(title or f"LineWorld (step {self._steps})", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frame = buf.reshape(h, w, 4)[:, :, :3].copy()
        plt.close(fig)
        return frame
