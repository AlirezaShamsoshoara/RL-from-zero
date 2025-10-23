from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from .envs import make as _make_env


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int, env_kwargs: Optional[Dict[str, Any]] = None):
    """
    Create and initialize environment.

    Args:
        env_id: Environment identifier
        seed: Random seed
        env_kwargs: Additional environment kwargs

    Returns:
        Initialized environment
    """
    env_kwargs = env_kwargs or {}
    env = _make_env(env_id, **env_kwargs)
    env.reset(seed=seed)
    return env


def save_checkpoint(
    path: str, q_tables: np.ndarray, step: int, best_return: float
) -> None:
    """
    Save Q-tables checkpoint to disk.

    Args:
        path: Path to save checkpoint
        q_tables: Q-tables array (shape: [n_agents, n_states, n_actions, n_actions])
        step: Training step/episode number
        best_return: Best average return achieved
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "q_tables": torch.from_numpy(np.asarray(q_tables, dtype=np.float32)),
        "step": step,
        "best_return": best_return,
    }
    torch.save(data, path)


def load_checkpoint(path: str):
    """
    Load checkpoint from disk.

    Args:
        path: Path to checkpoint file

    Returns:
        Checkpoint dictionary with keys: q_tables, step, best_return
    """
    return torch.load(path, map_location="cpu")
