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
    torch.cuda.manual_seed_all(seed)


def make_env(
    env_id: str,
    seed: int,
    render_mode: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode, **(env_kwargs or {}))
    try:
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
    except BaseException:
        env.close()
        raise
    return env


def save_checkpoint(
    path: str,
    agent: "DQNAgent",
    step: int,
    best_return: float,
    *,
    config: Optional[dict] = None,
):
    from deepQN.dqn.agent import DQNAgent

    if not isinstance(agent, DQNAgent):
        raise TypeError("agent must be an instance of DQNAgent")
    if not isinstance(step, (int, np.integer)) or isinstance(step, bool) or step < 0:
        raise ValueError("step must be a nonnegative integer")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = agent.state_dict()
    payload.update({"step": step, "best_return": best_return, "config": config})
    torch.save(payload, path)


def load_checkpoint(path: str, agent: "DQNAgent") -> Dict[str, Any]:
    from deepQN.dqn.agent import DQNAgent

    if not isinstance(agent, DQNAgent):
        raise TypeError("agent must be an instance of DQNAgent")
    data = torch.load(path, map_location="cpu", weights_only=True)
    agent.load_state_dict(data)
    return data
