"""Shared argument validation and discrete PPO checkpoint helpers."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from PPO.ppo.agent import PPOAgent
from PPO.ppo.config import Config
from PPO.ppo.utils import load_checkpoint


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def seed_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("must be between 0 and 2**32 - 1")
    return number


def validate_seed_range(
    parser: argparse.ArgumentParser, seed: int, episodes: int
) -> None:
    if seed + episodes > 2**32:
        parser.error("seed + episodes - 1 must be less than 2**32")


def validate_env(env: gym.Env) -> None:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("PPO helpers require a Box observation space")
    if len(env.observation_space.shape) != 1:
        raise ValueError("PPO helpers require flat observations")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("PPO helpers require a Discrete action space")
    if env.action_space.start != 0:
        raise ValueError("PPO helpers require zero-based discrete actions")


def build_agent(cfg: Config, env: gym.Env) -> PPOAgent:
    """Build the checkpoint architecture on CPU for repeatable evaluation."""
    validate_env(env)
    return PPOAgent(
        obs_dim=int(np.prod(env.observation_space.shape)),
        act_dim=int(env.action_space.n),
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        lr=cfg.learning_rate,
        clip_coef=cfg.clip_coef,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        device="cpu",
    )


def deterministic_action(agent: PPOAgent, obs: np.ndarray) -> int:
    actions, _, _ = agent.act(
        np.asarray(obs, dtype=np.float32)[None, :], deterministic=True
    )
    return int(np.asarray(actions).reshape(-1)[0])


def load_agent(cfg: Config, env: gym.Env, checkpoint: Path):
    agent = build_agent(cfg, env)
    payload = load_checkpoint(str(checkpoint), agent.model)
    agent.model.eval()
    return agent, payload


def checkpoint_metadata(path: Path, payload: dict) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    best_return = payload.get("best_return")
    if best_return is not None:
        best_return = float(best_return)
        if not math.isfinite(best_return):
            best_return = None
    step = payload.get("step")
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
        "step": int(step) if step is not None else None,
        "best_return": best_return,
    }
