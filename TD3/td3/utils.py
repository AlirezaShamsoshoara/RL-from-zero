from __future__ import annotations
import os
import random
from typing import Dict, Optional, Any
import gym
import numpy as np
import torch

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(
    env_id: str,
    seed: int,
    render_mode: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> gym.Env:
    env_kwargs = env_kwargs or {}
    env = gym.make(env_id, render_mode=render_mode, **env_kwargs)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    return env


def save_checkpoint(path: str, agent: "TD3Agent", step: int, best_return: float):
    from TD3.td3.agent import TD3Agent  # local import to avoid circular

    if not isinstance(agent, TD3Agent):
        raise TypeError("agent must be an instance of TD3Agent")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = agent.state_dict()
    payload["step"] = step
    payload["best_return"] = best_return
    torch.save(payload, path)


def load_checkpoint(path: str, agent: "TD3Agent") -> Dict[str, Any]:
    from TD3.td3.agent import TD3Agent  # local import to avoid circular

    if not isinstance(agent, TD3Agent):
        raise TypeError("agent must be an instance of TD3Agent")
    data = torch.load(path, map_location="cpu")
    agent.load_state_dict(data)
    return data
