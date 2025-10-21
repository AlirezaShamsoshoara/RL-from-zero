from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from .envs import make as _make_env


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int, env_kwargs: Optional[Dict[str, Any]] = None):
    env_kwargs = env_kwargs or {}
    env = _make_env(env_id, **env_kwargs)
    env.reset(seed=seed)
    return env


def save_checkpoint(path: str, q_tables: np.ndarray, step: int, best_return: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "q_tables": torch.from_numpy(np.asarray(q_tables, dtype=np.float32)),
        "step": step,
        "best_return": best_return,
    }
    torch.save(data, path)


def load_checkpoint(path: str):
    return torch.load(path, map_location="cpu")
