from __future__ import annotations

from typing import Any, Dict

from .grid_soccer import GridSoccerEnv
from .line_world import LineWorldEnv

ENV_REGISTRY: Dict[str, type] = {
    "line_world": LineWorldEnv,
    "nash_line_world": LineWorldEnv,
    "grid_soccer": GridSoccerEnv,
}


def make(env_id: str, **kwargs: Any):
    """
    Create environment by ID.

    Args:
        env_id: Environment identifier
        **kwargs: Environment-specific kwargs

    Returns:
        Environment instance

    Raises:
        ValueError: If env_id not found in registry
    """
    try:
        env_cls = ENV_REGISTRY[env_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown env_id '{env_id}'. Available: {list(ENV_REGISTRY)}"
        ) from exc
    return env_cls(**kwargs)


__all__ = ["make", "LineWorldEnv", "GridSoccerEnv", "ENV_REGISTRY"]
