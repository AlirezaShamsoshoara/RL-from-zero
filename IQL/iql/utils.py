from __future__ import annotations
import os
import random
from typing import Any, Dict, List, Optional
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


def save_checkpoint(path: str, agent: "IQLAgent", step: int, best_return: float):
    from IQL.iql.agent import IQLAgent  # local import to avoid circular dependency

    if not isinstance(agent, IQLAgent):
        raise TypeError("agent must be an instance of IQLAgent")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = agent.state_dict()
    payload["step"] = step
    payload["best_return"] = best_return
    torch.save(payload, path)


def load_checkpoint(path: str, agent: "IQLAgent") -> Dict[str, Any]:
    from IQL.iql.agent import IQLAgent  # local import to avoid circular dependency

    if not isinstance(agent, IQLAgent):
        raise TypeError("agent must be an instance of IQLAgent")
    data = torch.load(path, map_location="cpu")
    agent.load_state_dict(data)
    return data


def evaluate_policy(
    agent: "IQLAgent",
    env: gym.Env,
    episodes: int,
    seed: int,
) -> List[float]:
    returns: List[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_return += float(reward)
        returns.append(ep_return)
    return returns
