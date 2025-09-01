from __future__ import annotations
import os
import random
from typing import Optional, Dict, Any
import numpy as np
import torch
import gym


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int, render_mode: Optional[str] = None, env_kwargs: Optional[Dict[str, Any]] = None):
    env_kwargs = env_kwargs or {}
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
    env.reset(seed=seed)
    return env


def save_checkpoint(path: str, q_table: np.ndarray, step: int, best_return: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "q_table": torch.from_numpy(q_table),
        "step": step,
        "best_return": best_return,
    }, path)


def load_checkpoint(path: str):
    data = torch.load(path, map_location="cpu")
    return data

