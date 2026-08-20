from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch

from .envs import make as _make_env


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int, env_kwargs: Optional[Dict[str, Any]] = None):
    """
    Create and initialize environment.

    Args:
        env_id: Environment identifier
        seed: Random seed
        env_kwargs: Additional environment kwargs

    Returns:
        Initialized environment
    """
    env_kwargs = env_kwargs or {}
    env = _make_env(env_id, **env_kwargs)
    env.reset(seed=seed)
    return env


def save_checkpoint(
    path: str, q_tables: np.ndarray, step: int, best_return: float
) -> None:
    """
    Save Q-tables checkpoint to disk.

    Args:
        path: Path to save checkpoint
        q_tables: Q-tables array (shape: [n_agents, n_states, n_actions, n_actions])
        step: Training step/episode number
        best_return: Best average return achieved
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "q_tables": torch.from_numpy(np.asarray(q_tables, dtype=np.float32)),
        "step": step,
        "best_return": best_return,
    }
    torch.save(data, path)


def load_checkpoint(path: str):
    """
    Load checkpoint from disk.

    Args:
        path: Path to checkpoint file

    Returns:
        Checkpoint dictionary with keys: q_tables, step, best_return
    """
    return torch.load(path, map_location="cpu")


def evaluate_vs_random(agent, env, episodes: int, seed: int, max_steps: int = 60):
    """Evaluate the learned agent 0 (greedy Nash policy) against a random agent 1.

    On a zero-sum game a strong policy should win most games against a random
    opponent, which is a meaningful "best" signal for self-play (where the raw
    self-play return is ~0 at equilibrium).

    Returns:
        (win_rate, draw_rate) for the learned agent 0.
    """
    wins = 0
    draws = 0
    for ep in range(episodes):
        states = env.reset(seed=seed + ep)
        steps = 0
        scorer = None
        while steps < max_steps:
            # Against a known (random = uniform) opponent, play the best response
            # extracted from the learned Q, not the hedging equilibrium.
            a0 = agent.best_response_action(states[0], agent=0)
            a1 = random.randint(0, env.n_actions - 1)
            step_result = env.step([a0, a1])
            states = step_result.observations
            steps += 1
            if all(step_result.terminated):
                scorer = step_result.info.get("scorer")
                break
            if step_result.truncated:
                break
        if scorer == 0:
            wins += 1
        elif scorer is None:
            draws += 1
    return wins / episodes, draws / episodes
