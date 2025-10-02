from __future__ import annotations
import os
import random
from typing import Callable, Optional
import numpy as np
import torch
import gym


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int, render_mode: Optional[str] = None) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    return env


def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, best_return: float) -> None:
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


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
    data = torch.load(path, map_location="cpu")
    model.load_state_dict(data["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in data:
        optimizer.load_state_dict(data["optimizer_state_dict"])
    return data


def compute_returns(rewards: torch.Tensor, dones: torch.Tensor, last_value: torch.Tensor, gamma: float) -> torch.Tensor:
    R = last_value
    returns = torch.zeros_like(rewards)
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * (1.0 - dones[step])
        returns[step] = R
    return returns


class SharedAdam(torch.optim.Adam):
    """Adam optimizer with shared states across processes."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.zeros(1)
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
                state["step"].share_memory_()

    def share_memory(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()
                state["step"].share_memory_()
