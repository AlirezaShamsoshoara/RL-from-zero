from __future__ import annotations
import os
import random
from typing import Callable, Optional
import numpy as np
import torch
import gymnasium as gym


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AcrobotShapedReward(gym.Wrapper):
    """Potential-based reward shaping for Acrobot-v1.

    Acrobot's default reward (-1 per step) is too sparse for A3C to learn
    from random exploration alone.  This wrapper adds a potential-based
    bonus F = γ·Φ(s') − Φ(s) where Φ is the tip height.

    Potential-based shaping preserves the optimal policy (Ng et al., 1999)
    while providing an immediate, per-action learning signal: actions that
    swing the tip upward get higher reward, downward get lower.
    """

    def __init__(self, env: gym.Env, gamma: float = 0.99):
        super().__init__(env)
        self._gamma = gamma
        self._prev_height: float = -2.0  # bottom (default)

    @staticmethod
    def _tip_height(obs) -> float:
        """Tip height = −cos(θ₁) − cos(θ₁+θ₂), range [−2, 2]."""
        cos_sum = obs[0] * obs[2] - obs[1] * obs[3]  # cos(θ₁ + θ₂)
        return float(-obs[0] - cos_sum)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_height = self._tip_height(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_height = self._tip_height(obs)
        # F = γ·Φ(s') − Φ(s)
        shaping = self._gamma * new_height - self._prev_height
        self._prev_height = new_height
        return obs, reward + shaping, terminated, truncated, info


class RewardScaler(gym.RewardWrapper):
    """Multiply rewards by a constant scale factor."""

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self._scale = scale

    def reward(self, reward):
        return reward * self._scale


def _wrap_env(env: gym.Env, env_id: str, reward_shaping: bool, gamma: float = 0.99, reward_scale: float = 1.0) -> gym.Env:
    if reward_shaping and "Acrobot" in env_id:
        env = AcrobotShapedReward(env, gamma=gamma)
    if reward_scale != 1.0:
        env = RewardScaler(env, scale=reward_scale)
    return env


def make_env(
    env_id: str,
    seed: int,
    render_mode: Optional[str] = None,
    reward_shaping: bool = False,
    gamma: float = 0.99,
    reward_scale: float = 1.0,
) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    env = _wrap_env(env, env_id, reward_shaping, gamma=gamma, reward_scale=reward_scale)
    env.reset(seed=seed)
    return env


def make_vec_env(
    env_id: str,
    num_envs: int,
    seed: int,
    reward_shaping: bool = False,
    gamma: float = 0.99,
    reward_scale: float = 1.0,
) -> gym.vector.VectorEnv:
    from gymnasium.vector import AutoresetMode

    def _make_single(rank: int):
        def _thunk():
            env = gym.make(env_id)
            env = _wrap_env(env, env_id, reward_shaping, gamma=gamma, reward_scale=reward_scale)
            env.reset(seed=seed + rank)
            return env
        return _thunk

    return gym.vector.SyncVectorEnv(
        [_make_single(i) for i in range(num_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )


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
