from __future__ import annotations

import numpy as np


class ActionHoldExplorer:
    """Temporally correlated epsilon exploration; inference remains greedy DQN."""

    def __init__(self, hold_min: int = 1, hold_max: int = 1):
        if (
            any(
                not isinstance(v, int) or isinstance(v, bool) or v <= 0
                for v in (hold_min, hold_max)
            )
            or hold_max < hold_min
        ):
            raise ValueError(
                "exploration hold bounds must be ordered positive integers"
            )
        self.hold_min, self.hold_max = hold_min, hold_max
        self.mean_hold = (hold_min + hold_max) / 2.0
        self.remaining = 0
        self.action = None

    def start_probability(self, epsilon: float) -> float:
        if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be finite and in [0, 1]")
        # A pulse lasts H steps while a greedy decision lasts one. Correct the
        # start probability so epsilon approximates the random STEP fraction.
        return epsilon / (self.mean_hold * (1.0 - epsilon) + epsilon)

    def act(self, agent, obs, epsilon: float):
        if self.hold_max == 1:
            # Preserve the baseline's exact selection/RNG behavior.
            return agent.act(obs, epsilon=epsilon), None
        if self.remaining:
            agent._observation(obs)
            agent.step_count += 1
            self.remaining -= 1
            return self.action, True
        if np.random.rand() < self.start_probability(epsilon):
            self.action = agent.act(obs, epsilon=1.0)
            self.remaining = (
                int(np.random.randint(self.hold_min, self.hold_max + 1)) - 1
            )
            return self.action, True
        return agent.act(obs, epsilon=0.0), False

    def reset(self):
        self.remaining = 0
        self.action = None
