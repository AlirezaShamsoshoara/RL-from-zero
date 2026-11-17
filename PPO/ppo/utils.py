from __future__ import annotations
import os
import random
from typing import Callable, Optional
import numpy as np
import torch
import gymnasium as gym


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_vec_env(
    env_id: str, num_envs: int, seed: int, render_mode: Optional[str] = None
):
    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _thunk():
            env = gym.make(env_id, render_mode=render_mode if num_envs == 1 else None)
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return env

        return _thunk

    # Modern Gymnasium otherwise inserts a reset-only vector step. SAME_STEP
    # returns final observations in info and keeps every counted step a transition.
    kwargs = {}
    if hasattr(gym.vector, "AutoresetMode"):
        kwargs["autoreset_mode"] = gym.vector.AutoresetMode.SAME_STEP
    elif int(gym.__version__.split(".")[0]) >= 1:
        raise RuntimeError(
            "PPO requires Gymnasium 1.1+ for explicit same-step autoreset, "
            "or the legacy 0.29 API; Gymnasium 1.0 is not supported."
        )
    if num_envs == 1:
        return gym.vector.SyncVectorEnv([make_env(0)], **kwargs)
    return gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)], **kwargs)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_return: float,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_return": best_return,
        },
        path,
    )


def load_checkpoint(
    path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
):
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in data:
        optimizer.load_state_dict(data["optimizer_state_dict"])
    return data
