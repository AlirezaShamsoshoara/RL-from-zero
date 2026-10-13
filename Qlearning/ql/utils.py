from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(
    env_id: str,
    seed: int,
    render_mode: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
):
    env = gym.make(env_id, render_mode=render_mode, **(env_kwargs or {}))
    try:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    except BaseException:
        env.close()
        raise
    return env


def save_checkpoint(
    path: str,
    q_table: np.ndarray,
    step: int,
    best_return: float,
    *,
    global_step: int = 0,
    config: Optional[dict] = None,
):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    table = np.asarray(q_table, dtype=np.float32)
    if table.ndim != 2 or not np.all(np.isfinite(table)):
        raise ValueError("checkpoint Q-table must be a finite two-dimensional array")
    torch.save(
        {
            "q_table": torch.from_numpy(table.copy()),
            "step": int(step),
            "global_step": int(global_step),
            "best_return": float(best_return),
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or "q_table" not in data:
        raise ValueError("checkpoint must contain a Q-table")
    return data
