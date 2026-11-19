"""Shared validation, checkpoint loading, and capped tabular-policy rollouts."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from Qlearning.ql.agent import QLearningAgent
from Qlearning.ql.config import Config
from Qlearning.ql.utils import load_checkpoint


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
    for name, space in (
        ("observation", env.observation_space),
        ("action", env.action_space),
    ):
        if not isinstance(space, gym.spaces.Discrete):
            raise TypeError(f"Qlearning helpers require a Discrete {name} space")
        if space.start != 0:
            raise ValueError(f"Qlearning helpers require a zero-based {name} space")


def validate_config(cfg: Config) -> None:
    if not isinstance(cfg.max_steps_per_episode, int) or cfg.max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be a positive integer")
    if not math.isfinite(cfg.gamma) or not 0 <= cfg.gamma <= 1:
        raise ValueError("gamma must be finite and between 0 and 1")


def build_agent(cfg: Config, env: gym.Env) -> QLearningAgent:
    validate_env(env)
    return QLearningAgent(
        n_states=int(env.observation_space.n),
        n_actions=int(env.action_space.n),
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
    )


def load_agent(cfg: Config, env: gym.Env, checkpoint: Path):
    """Load tensor or NumPy Q tables; no neural network or optimizer is involved."""
    agent = build_agent(cfg, env)
    payload = load_checkpoint(str(checkpoint))
    q_table = payload["q_table"]
    if hasattr(q_table, "detach"):
        q_table = q_table.detach().cpu().numpy()
    q_table = np.asarray(q_table, dtype=np.float32)
    expected = (env.observation_space.n, env.action_space.n)
    if q_table.shape != expected:
        raise ValueError(
            f"Checkpoint Q shape {q_table.shape} does not match env {expected}"
        )
    if not np.isfinite(q_table).all():
        raise ValueError("Checkpoint contains non-finite Q values")
    global_step = payload.get("global_step", 0)
    agent.load_state_dict({"q_table": q_table.copy(), "global_step": global_step})
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
    global_step = payload.get("global_step")
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
        "step": int(step) if step is not None else None,
        "step_units": "training episode",
        "global_step": int(global_step) if global_step is not None else None,
        "global_step_units": "environment transitions",
        "best_return": best_return,
        "q_shape": list(payload["q_table"].shape),
    }


def _state(env: gym.Env, observation) -> int:
    if not isinstance(observation, (int, np.integer)):
        raise TypeError("Discrete environment returned a non-integer state")
    state = int(observation)
    if not env.observation_space.contains(state):
        raise ValueError(f"State {state} is outside the observation space")
    return state


def rollout_episode(
    env: gym.Env,
    agent: QLearningAgent | None,
    seed: int,
    max_steps: int,
    gamma: float,
    random_policy: bool = False,
    capture_frames: bool = False,
) -> dict:
    """Evaluate without learning or changing the agent's step counter."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    env.action_space.seed(seed)
    observation, _ = env.reset(seed=seed)
    state = _state(env, observation)
    start_state = state
    start_value = float(np.max(agent.Q[state])) if agent is not None else None
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
            else agent.act(state, deterministic=True)
        )
        observation, reward, terminated, truncated, _ = env.step(int(action))
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("Environment returned a non-finite reward")
        total += reward
        discounted += discount * reward
        discount *= gamma
        length += 1
        state = _state(env, observation)
        if terminated or truncated:
            break
    if not math.isfinite(total) or not math.isfinite(discounted):
        raise ValueError("Episode return is non-finite")
    if capture_frames:
        capture()  # Include the reached goal, hole, or capped final state.
    return {
        "seed": seed,
        "start_state": start_state,
        "start_value": start_value,
        "return": total,
        "discounted_return": discounted,
        "length": length,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "capped": bool(length == max_steps and not (terminated or truncated)),
        "frames": frames,
    }
