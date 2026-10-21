"""DQN helper validation, checkpoint construction, and native-reward rollouts."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from deepQN.dqn.agent import DQNAgent
from deepQN.dqn.config import Config
from deepQN.dqn.utils import load_checkpoint


def redact_config(value):
    """Keep checkpoint/evaluation provenance without retaining credential fields."""
    secrets = {
        "wandb_key",
        "wandb_api_key",
        "api_key",
        "password",
        "token",
        "access_token",
        "secret",
    }
    if isinstance(value, dict):
        return {
            key: redact_config(item)
            for key, item in value.items()
            if str(key).lower() not in secrets
        }
    if isinstance(value, (list, tuple)):
        return [redact_config(item) for item in value]
    return value


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
        raise TypeError("DQN helpers require a Box observation space")
    if len(env.observation_space.shape) != 1:
        raise ValueError("DQN helpers require flat observations")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise TypeError("DQN helpers require a Discrete action space")
    if env.action_space.start != 0:
        raise ValueError("DQN helpers require zero-based discrete actions")


def episode_cap(cfg: Config) -> int:
    cap = getattr(cfg, "max_steps_per_episode", 1000)
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError("max_steps_per_episode must be a positive integer")
    if not math.isfinite(cfg.gamma) or not 0 <= cfg.gamma <= 1:
        raise ValueError("gamma must be finite and between 0 and 1")
    return cap


def build_agent(cfg: Config, env: gym.Env) -> DQNAgent:
    validate_env(env)
    return DQNAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        lr=cfg.lr,
        gamma=cfg.gamma,
        target_update_interval=cfg.target_update_interval,
        max_grad_norm=cfg.max_grad_norm,
        double_dqn=cfg.double_dqn,
        device="cpu",
        normalize_observations=cfg.normalize_observations,
    )


def load_agent(cfg: Config, env: gym.Env, checkpoint: Path):
    agent = build_agent(cfg, env)
    payload = load_checkpoint(str(checkpoint), agent)
    agent.q_network.eval()
    agent.target_q_network.eval()
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
    metadata = {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
        "best_return": best_return,
        "step_units": "training environment transitions",
        "normalize_observations": payload.get("normalize_observations", False),
        "observation_low": payload.get("observation_low"),
        "observation_high": payload.get("observation_high"),
    }
    for key in ("step", "step_count", "train_steps"):
        value = payload.get(key)
        if value is not None and (
            not isinstance(value, (int, np.integer))
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(f"{key} must be a nonnegative integer")
        metadata[key] = int(value) if value is not None else None
    return metadata


def observation(env: gym.Env, obs) -> np.ndarray:
    array = np.asarray(obs, dtype=np.float32)
    if array.shape != env.observation_space.shape or not np.isfinite(array).all():
        raise ValueError(
            "Environment returned non-finite or incorrectly shaped observations"
        )
    return array.copy()


def rollout_episode(
    env: gym.Env,
    agent: DQNAgent | None,
    seed: int,
    max_steps: int,
    gamma: float,
    env_id: str,
    random_policy: bool = False,
    capture_frames: bool = False,
) -> dict:
    """No learning, reward shaping, or agent counter updates during evaluation."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)
    obs = observation(env, obs)
    initial_obs = obs.tolist()
    initial_q = float(agent.q_value(obs)) if agent is not None else None
    if initial_q is not None and not math.isfinite(initial_q):
        raise ValueError("Initial Q estimate is non-finite")
    mountaincar = env_id == "MountainCar-v0"
    max_position = float(obs[0]) if mountaincar else None
    total = 0.0
    discounted = 0.0
    discount = 1.0
    frames = []
    terminated = False
    truncated = False
    length = 0

    def capture():
        frame = env.render()
        if frame is None:
            raise RuntimeError("rgb_array rendering returned no frame")
        frames.append(np.asarray(frame).copy())

    for _ in range(max_steps):
        if capture_frames:
            capture()
        action = (
            env.action_space.sample()
            if random_policy
            else agent.act(obs, epsilon=0.0, deterministic=True)
        )
        next_obs, reward, terminated, truncated, _ = env.step(int(action))
        obs = observation(env, next_obs)
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("Environment returned a non-finite reward")
        total += reward
        discounted += discount * reward
        discount *= gamma
        length += 1
        if mountaincar:
            max_position = max(max_position, float(obs[0]))
        if terminated or truncated:
            break
    if not math.isfinite(total) or not math.isfinite(discounted):
        raise ValueError("Episode return is non-finite")
    if capture_frames:
        capture()
    return {
        "seed": seed,
        "initial_observation": initial_obs,
        "initial_q": initial_q,
        "return": total,
        "discounted_return": discounted,
        "length": length,
        "max_position": max_position,
        "success": bool(terminated) if mountaincar else None,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "capped": bool(length == max_steps and not (terminated or truncated)),
        "frames": frames,
    }
