from __future__ import annotations
import os
import random
from typing import Any, Callable, Dict, Optional
import gymnasium as gym
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_vec_env(
    env_id: str,
    num_envs: int,
    seed: int,
    render_mode: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
):
    env_kwargs = env_kwargs or {}

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _thunk():
            kwargs = dict(env_kwargs)
            render = render_mode if num_envs == 1 else None
            env = gym.make(env_id, render_mode=render, **kwargs)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=seed + rank)
            return env

        return _thunk

    if num_envs == 1:
        return gym.vector.SyncVectorEnv([make_env(0)])
    return gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)])


def save_checkpoint(path: str, agent: "TRPOAgent", step: int, best_return: float):
    from TRPO.trpo.agent import TRPOAgent  # local import to avoid circular deps

    if not isinstance(agent, TRPOAgent):
        raise TypeError("agent must be an instance of TRPOAgent")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = agent.state_dict()
    payload["step"] = step
    payload["best_return"] = best_return
    torch.save(payload, path)


def load_checkpoint(path: str, agent: "TRPOAgent"):
    from TRPO.trpo.agent import TRPOAgent  # local import

    if not isinstance(agent, TRPOAgent):
        raise TypeError("agent must be an instance of TRPOAgent")
    data = torch.load(path, map_location="cpu")
    agent.load_state_dict(data)
    return data

