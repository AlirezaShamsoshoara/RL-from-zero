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
    info: Dict[str, object]


class GridSoccerEnv:
    """
    Two-player zero-sum grid soccer (a small Markov game).

    A compact version of Littman's (1994) soccer used to study minimax / Nash
    Q-learning. Two players occupy a `rows x cols` grid; one holds the ball.
    Agent 0 attacks the RIGHT goal (column cols-1), agent 1 attacks the LEFT goal
    (column 0); a goal spans the middle rows of that column. Carrying the ball
    into your goal scores (+1 to the scorer, -1 to the other): the game is
    zero-sum, so the maxmin (security) strategy is the exact Nash equilibrium,
    which is what the LP solver in this package computes.

    Steal mechanic (Littman): the two moves are applied in a random order; if a
    player tries to move into the cell currently occupied by the other player, it
    does not move, and if it was carrying the ball the ball transfers to the
    stationary player. This is how defense / interception happens.

    State: (pos0, pos1, ball_owner) flattened to a single shared index that BOTH
    agents observe (Nash Q-learning indexes both stage-game payoff matrices at the
    same state).
    """

    # 0=stay, 1=up, 2=down, 3=left, 4=right
    ACTION_MEANINGS = {0: "stay", 1: "up", 2: "down", 3: "left", 4: "right"}
    _DELTAS = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    _PALETTE = ["#2563eb", "#dc2626"]

    def __init__(
        self,
        rows: int = 4,
        cols: int = 5,
        max_steps: int = 40,
        goal_reward: float = 1.0,
        shaping: float = 0.1,
    ):
        if rows < 2 or cols < 3:
            raise ValueError("grid soccer needs rows >= 2 and cols >= 3")
        self.n_agents = 2
        self.rows = rows
        self.cols = cols
        self.max_steps = max_steps
        self.goal_reward = goal_reward
        # Zero-sum dense shaping: the ball carrier is rewarded for advancing the
        # ball toward its own goal (and the opponent gets the negation), so the
        # game stays zero-sum but there is signal before any goal is scored.
        self.shaping = shaping

        self.n_cells = rows * cols
        self.n_actions = 5
        # Shared joint state: (pos0 * n_cells + pos1) * 2 + ball_owner.
        self.n_states = self.n_cells * self.n_cells * 2

        # Goal cells span the middle rows of each side column.
        if rows >= 2:
            mid = rows // 2
            self._goal_rows = {mid - 1, mid} if rows % 2 == 0 else {mid}
        self.left_goal_col = 0
        self.right_goal_col = cols - 1

        self._rng = random.Random()
        self.positions: List[Tuple[int, int]] = []
        self.ball_owner = 0
        self._steps = 0

    # -- helpers ---------------------------------------------------------

    def _rc(self, p: int) -> Tuple[int, int]:
        return divmod(p, self.cols)

    def _idx(self, rc: Tuple[int, int]) -> int:
        return rc[0] * self.cols + rc[1]

    def _encode(self) -> int:
        p0 = self._idx(self.positions[0])
        p1 = self._idx(self.positions[1])
        return (p0 * self.n_cells + p1) * 2 + self.ball_owner

    def _encode_all(self) -> List[int]:
        code = self._encode()
        return [code, code]

    def encode(self, positions: Sequence[Tuple[int, int]], ball_owner: int) -> int:
        """Encode (pos0, pos1, ball_owner) into the shared joint-state index."""
        return (self._idx(positions[0]) * self.n_cells + self._idx(positions[1])) * 2 + ball_owner

    def decode(self, state: int) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
        """Inverse of encode: state -> (pos0, pos1, ball_owner)."""
        ball_owner = state % 2
        pair = state // 2
        p1 = pair % self.n_cells
        p0 = pair // self.n_cells
        return self._rc(p0), self._rc(p1), ball_owner

    def _is_goal_cell(self, agent: int, rc: Tuple[int, int]) -> bool:
        r, c = rc
        if r not in self._goal_rows:
            return False
        return c == self.right_goal_col if agent == 0 else c == self.left_goal_col

    def _ball_dist_to_goal_from(self, positions: Sequence[Tuple[int, int]], owner: int) -> int:
        target = self.right_goal_col if owner == 0 else self.left_goal_col
        return abs(positions[owner][1] - target)

    def _ball_dist_to_goal(self, owner: int) -> int:
        return self._ball_dist_to_goal_from(self.positions, owner)

    def simulate(
        self,
        state: int,
        actions: Sequence[int],
        order: Tuple[int, int],
        shaping: Optional[float] = None,
    ) -> Tuple[int, float, float, bool, Optional[int]]:
        """Deterministic dynamics: (state, joint action, move order) -> outcome.

        Pure function (does not mutate env). Returns
        ``(next_state, r0, r1, terminal, scorer)`` where ``r0 + r1 == 0`` (zero-sum).
        ``shaping`` defaults to the env's shaping coefficient; pass ``0.0`` to
        get the "true" +/-1 game reward that the exact solver operates on.
        """
        if shaping is None:
            shaping = self.shaping
        pos0, pos1, ball_owner = self.decode(state)
        positions = [list(pos0), list(pos1)]
        prev_owner = ball_owner
        prev_dist = self._ball_dist_to_goal_from([tuple(positions[0]), tuple(positions[1])], prev_owner)

        for agent in order:
            dr, dc = self._DELTAS[int(actions[agent])]
            r, c = positions[agent]
            nr = min(max(r + dr, 0), self.rows - 1)
            nc = min(max(c + dc, 0), self.cols - 1)
            other = 1 - agent
            if (nr, nc) == tuple(positions[other]):
                if ball_owner == agent:
                    ball_owner = other
            else:
                positions[agent] = [nr, nc]

        r0 = 0.0
        if shaping:
            owner = ball_owner
            if owner == prev_owner:
                delta = prev_dist - self._ball_dist_to_goal_from(
                    [tuple(positions[0]), tuple(positions[1])], owner
                )
            else:
                delta = 1.0  # possession won this step
            # Written from agent 0's perspective: +delta if 0 has the ball, else -delta.
            r0 = shaping * delta * (1.0 if owner == 0 else -1.0)

        scorer = None
        if self._is_goal_cell(0, tuple(positions[0])) and ball_owner == 0:
            scorer = 0
        elif self._is_goal_cell(1, tuple(positions[1])) and ball_owner == 1:
            scorer = 1
        terminal = scorer is not None
        if terminal:
            r0 += self.goal_reward if scorer == 0 else -self.goal_reward

        next_state = self.encode(
            [tuple(positions[0]), tuple(positions[1])], ball_owner
        )
        return next_state, r0, -r0, terminal, scorer

    # -- API -------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> List[int]:
        if seed is not None:
            self._rng.seed(seed)
        self._steps = 0
        mid = self.rows // 2
        # Start on opposite sides, off the goal columns, not overlapping.
        self.positions = [(mid, 1), (mid, self.cols - 2)]
        if self.positions[0] == self.positions[1]:
            self.positions = [(0, 1), (self.rows - 1, self.cols - 2)]
        self.ball_owner = self._rng.randint(0, 1)
        return self._encode_all()

    def step(self, actions: Sequence[int]) -> StepResult:
        if len(actions) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {len(actions)}")

        self._steps += 1
        order = [0, 1]
        self._rng.shuffle(order)

        state = self._encode()
        next_state, r0, r1, terminal, scorer = self.simulate(state, actions, tuple(order))
        (pos0, pos1, self.ball_owner) = self.decode(next_state)
        self.positions = [pos0, pos1]

        terminated = [terminal, terminal]
        rewards = [r0, r1]
        truncated = self._steps >= self.max_steps
        info = {"positions": tuple(self.positions), "ball_owner": self.ball_owner,
                "scorer": scorer}
        return StepResult(
            observations=self._encode_all(),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def action_meanings(self) -> Dict[int, str]:
        return dict(self.ACTION_MEANINGS)

    def render(self, title: Optional[str] = None):
        """Render the board to an RGB numpy array (H, W, 3) via matplotlib."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(1.1 * self.cols + 1.5, 1.1 * self.rows), dpi=80)
        # Cells.
        for r in range(self.rows):
            for c in range(self.cols):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0,
                                           fill=False, edgecolor="#dddddd", lw=1.0))
        # Goals (shaded) on the middle rows of each side.
        for r in self._goal_rows:
            ax.add_patch(plt.Rectangle((self.left_goal_col - 0.5, r - 0.5), 1.0, 1.0,
                                       color=self._PALETTE[1], alpha=0.15))
            ax.add_patch(plt.Rectangle((self.right_goal_col - 0.5, r - 0.5), 1.0, 1.0,
                                       color=self._PALETTE[0], alpha=0.15))
        # Players; the ball carrier gets a black ring + a ball marker.
        for a in range(self.n_agents):
            r, c = self.positions[a]
            color = self._PALETTE[a]
            has_ball = self.ball_owner == a
            ax.scatter([c], [r], s=520, color=color,
                       edgecolors="black" if has_ball else color, linewidths=2.5, zorder=3)
            ax.text(c, r, f"A{a}", ha="center", va="center", color="white",
                    fontsize=10, zorder=4)
            if has_ball:
                ax.scatter([c + 0.28], [r - 0.28], s=90, color="white",
                           edgecolors="black", linewidths=1.2, zorder=5)
        ax.set_xlim(-0.6, self.cols - 0.4)
        ax.set_ylim(self.rows - 0.4, -0.6)  # row 0 at top
        ax.set_aspect("equal")
        ax.set_title(title or f"Grid Soccer (step {self._steps})", fontsize=11)
        ax.axis("off")
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frame = buf.reshape(h, w, 4)[:, :, :3].copy()
        plt.close(fig)
        return frame
