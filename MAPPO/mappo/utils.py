from __future__ import annotations
import os
import random
from typing import Optional, Dict, List
import numpy as np
import torch
from gymnasium import spaces
from pettingzoo.sisl import multiwalker_v9


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_action_grid(act_space: spaces.Box, bins: int) -> np.ndarray:
    low = np.asarray(act_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(act_space.high, dtype=np.float32).reshape(-1)
    axes = [np.linspace(lo, hi, bins).astype(np.float32) for lo, hi in zip(low, high)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], axis=1)


class DiscreteActionWrapper:
    def __init__(self, env, bins: int = 3):
        self.env = env
        self.possible_agents = getattr(env, "possible_agents", [])
        if not self.possible_agents:
            raise ValueError("Cannot discretize actions without possible_agents.")

        if bins < 2:
            raise ValueError("action_bins must be >= 2.")

        first_agent = self.possible_agents[0]
        act_space = env.action_space(first_agent)
        if not isinstance(act_space, spaces.Box):
            raise ValueError("DiscreteActionWrapper expects a Box action space.")
        if act_space.shape is None or len(act_space.shape) != 1:
            raise ValueError("Only 1D Box action spaces are supported for discretization.")

        self._action_vectors = _build_action_grid(act_space, bins)
        self._action_space = spaces.Discrete(self._action_vectors.shape[0])

    def action_space(self, agent_id):
        return self._action_space

    def observation_space(self, agent_id):
        return self.env.observation_space(agent_id)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, actions):
        mapped = {agent_id: self._map_action(action) for agent_id, action in actions.items()}
        return self.env.step(mapped)

    def close(self):
        return self.env.close()

    def _map_action(self, action):
        if isinstance(action, (int, np.integer)):
            idx = int(action)
        elif isinstance(action, np.ndarray) and action.shape == ():
            idx = int(action.item())
        else:
            action_arr = np.asarray(action, dtype=np.float32)
            if action_arr.shape == self._action_vectors.shape[1:]:
                return action_arr
            raise ValueError("Expected discrete action index or action vector.")

        if idx < 0 or idx >= self._action_vectors.shape[0]:
            raise ValueError("Discrete action index out of range.")
        return self._action_vectors[idx]

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_multiwalker_env(
    n_walkers: int = 3,
    seed: int = 42,
    render_mode: Optional[str] = None,
    discretize_actions: bool = True,
    action_bins: int = 3,
):
    """
    Create a MultiWalker environment from PettingZoo.
    Args:
        n_walkers: Number of walker agents (default: 3)
        seed: Random seed
        render_mode: Render mode ('human' for visualization, None for training)
        discretize_actions: Discretize continuous actions for categorical policy
        action_bins: Bins per action dimension when discretizing
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
    if discretize_actions:
        first_agent = env.possible_agents[0]
        act_space = env.action_space(first_agent)
        if isinstance(act_space, spaces.Box):
            env = DiscreteActionWrapper(env, bins=action_bins)
    env.reset(seed=seed)
    return env


def get_space_dims(env, return_action_space: bool = False):
    """
    Get observation and action space dimensions for multi-agent env.
    Returns:
        obs_dim: Observation dimension for single agent
        act_dim: Action dimension for single agent
        state_dim: Full state dimension (concatenated observations)
        n_agents: Number of agents
        act_space: Action space (if return_action_space=True)
    """
    agent_ids = env.possible_agents
    n_agents = len(agent_ids)

    # Get dimensions from first agent (assumes homogeneous agents)
    first_agent = agent_ids[0]
    obs_space = env.observation_space(first_agent)
    act_space = env.action_space(first_agent)

    obs_dim = obs_space.shape[0]
    if hasattr(act_space, "n"):
        act_dim = act_space.n
    elif isinstance(act_space, spaces.Box):
        if act_space.shape is None or len(act_space.shape) != 1:
            raise ValueError("Only 1D Box action spaces are supported.")
        act_dim = act_space.shape[0]
    else:
        raise ValueError("Unsupported action space type.")

    # State dimension is concatenation of all agent observations
    state_dim = obs_dim * n_agents

    if return_action_space:
        return obs_dim, act_dim, state_dim, n_agents, act_space
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
