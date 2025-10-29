from __future__ import annotations
import os
import random
from typing import Optional, Dict, List
import numpy as np
import torch
from pettingzoo.sisl import multiwalker_v9


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_multiwalker_env(n_walkers: int = 3, seed: int = 42, render_mode: Optional[str] = None):
    """
    Create a MultiWalker environment from PettingZoo.
    Args:
        n_walkers: Number of walker agents (default: 3)
        seed: Random seed
        render_mode: Render mode ('human' for visualization, None for training)
    Returns:
        env: PettingZoo parallel environment
    """
    env = multiwalker_v9.parallel_env(
        n_walkers=n_walkers,
        position_noise=1e-3,
        angle_noise=1e-3,
        forward_reward=1.0,
        terminate_reward=-100.0,
        fall_reward=-10.0,
        shared_reward=True,
        terminate_on_fall=True,
        remove_on_fall=True,
        max_cycles=500,
        render_mode=render_mode,
    )
    env.reset(seed=seed)
    return env


def get_space_dims(env):
    """
    Get observation and action space dimensions for multi-agent env.
    Returns:
        obs_dim: Observation dimension for single agent
        act_dim: Action dimension for single agent
        state_dim: Full state dimension (concatenated observations)
        n_agents: Number of agents
    """
    agent_ids = env.possible_agents
    n_agents = len(agent_ids)

    # Get dimensions from first agent (assumes homogeneous agents)
    first_agent = agent_ids[0]
    obs_space = env.observation_space(first_agent)
    act_space = env.action_space(first_agent)

    obs_dim = obs_space.shape[0]
    act_dim = act_space.n

    # State dimension is concatenation of all agent observations
    state_dim = obs_dim * n_agents

    return obs_dim, act_dim, state_dim, n_agents


def save_checkpoint(
    path: str,
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    step: int,
    best_return: float,
    share_policy: bool = False,
):
    """
    Save checkpoint for MAPPO agents.
    Args:
        path: Path to save checkpoint
        models: List of agent models
        optimizers: List of optimizers
        step: Current training step
        best_return: Best average return so far
        share_policy: Whether agents share policy (affects how we save)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if share_policy:
        # Only save one model since they're shared
        checkpoint = {
            "model_state_dict": models[0].state_dict(),
            "optimizer_state_dict": optimizers[0].state_dict(),
            "step": step,
            "best_return": best_return,
            "share_policy": True,
        }
    else:
        # Save all agent models separately
        checkpoint = {
            "model_state_dicts": [model.state_dict() for model in models],
            "optimizer_state_dicts": [opt.state_dict() for opt in optimizers],
            "step": step,
            "best_return": best_return,
            "share_policy": False,
        }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    models: List[torch.nn.Module],
    optimizers: Optional[List[torch.optim.Optimizer]] = None,
):
    """
    Load checkpoint for MAPPO agents.
    Args:
        path: Path to checkpoint
        models: List of agent models to load into
        optimizers: Optional list of optimizers to load into
    Returns:
        checkpoint data dictionary
    """
    data = torch.load(path, map_location="cpu")

    if data.get("share_policy", False):
        # Load shared policy into all models
        for model in models:
            model.load_state_dict(data["model_state_dict"])
        if optimizers is not None:
            optimizers[0].load_state_dict(data["optimizer_state_dict"])
    else:
        # Load individual policies
        for i, model in enumerate(models):
            model.load_state_dict(data["model_state_dicts"][i])
        if optimizers is not None:
            for i, opt in enumerate(optimizers):
                opt.load_state_dict(data["optimizer_state_dicts"][i])

    return data
