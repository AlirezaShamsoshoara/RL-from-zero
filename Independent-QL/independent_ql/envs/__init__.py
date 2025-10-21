from __future__ import annotations

from typing import Any, Dict, Type

from .line_world import LineWorldEnv

ENV_REGISTRY: Dict[str, Type[LineWorldEnv]] = {
    "line_world": LineWorldEnv,
    "independent_line_world": LineWorldEnv,
}


def make(env_id: str, **kwargs: Any) -> LineWorldEnv:
    try:
        env_cls = ENV_REGISTRY[env_id]
    except KeyError as exc:
        raise ValueError(f"Unknown env_id '{env_id}'. Available: {list(ENV_REGISTRY)}") from exc
    return env_cls(**kwargs)


__all__ = ["make", "LineWorldEnv", "ENV_REGISTRY"]
