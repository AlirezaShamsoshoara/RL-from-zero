from __future__ import annotations

import math
import os
import random
import tempfile
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch

if TYPE_CHECKING:
    from SAC.sac.agent import SACAgent


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
    kwargs = dict(env_kwargs or {})
    if "render_mode" in kwargs:
        raise ValueError("render_mode is a separate factory argument")
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    try:
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    except BaseException:
        env.close()
        raise


def _clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(item) for item in value)
    return value


def _metadata(value):
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            name = str(key).lower()
            if any(
                part in name
                for part in (
                    "wandb_key",
                    "api_key",
                    "password",
                    "secret",
                    "token",
                    "credential",
                )
            ):
                continue
            result[key] = _metadata(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("unsupported checkpoint metadata")


def _scalars(step, best_return):
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    if best_return is not None:
        if (
            isinstance(best_return, bool)
            or not isinstance(best_return, Real)
            or math.isnan(float(best_return))
            or best_return == math.inf
        ):
            raise ValueError(
                "best_return must be finite, null, or a legacy negative-infinity sentinel"
            )
    return int(step), None if best_return is None else float(best_return)


def save_checkpoint(
    path: str,
    agent: "SACAgent",
    step: int,
    best_return: Optional[float],
    config=None,
    *,
    metadata=None,
):
    from SAC.sac.agent import SACAgent

    if not isinstance(agent, SACAgent):
        raise TypeError("agent must be an instance of SACAgent")
    step, best_return = _scalars(step, best_return)
    payload = _clone(agent.state_dict())
    payload.update(step=step, best_return=best_return)
    if config is not None:
        payload["config"] = _metadata(config)
    if metadata is not None:
        payload["metadata"] = _metadata(metadata)
    destination = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str, agent: "SACAgent") -> Dict[str, Any]:
    from SAC.sac.agent import SACAgent

    if not isinstance(agent, SACAgent):
        raise TypeError("agent must be an instance of SACAgent")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict):
        raise ValueError("checkpoint must contain a mapping")
    _scalars(data.get("step", 0), data.get("best_return"))
    agent.load_state_dict(data)
    return data
