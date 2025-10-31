from __future__ import annotations
import os
import random
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int, n_agents: int = 3, max_cycles: int = 25, render_mode: Optional[str] = None):
    """
    Create a PettingZoo environment.

    Args:
        env_id: Environment ID (e.g., 'simple_spread_v3')
        seed: Random seed
        n_agents: Number of agents
        max_cycles: Maximum number of cycles per episode
        render_mode: Render mode ('human' for visualization, None for training)

    Returns:
        env: PettingZoo parallel environment
    """
    from pettingzoo.mpe import simple_spread_v3

    env = simple_spread_v3.parallel_env(
        N=n_agents,
        local_ratio=0.5,
        max_cycles=max_cycles,
        continuous_actions=True,
        render_mode=render_mode,
    )
    env.reset(seed=seed)
    return env


def get_space_info(env) -> Tuple[int, List[int], List[int], List[np.ndarray], List[np.ndarray]]:
    """
    Extract space information from PettingZoo environment.

    Args:
        env: PettingZoo parallel environment

    Returns:
        n_agents: Number of agents
        obs_dims: List of observation dimensions for each agent
        act_dims: List of action dimensions for each agent
        action_lows: List of action lower bounds for each agent
        action_highs: List of action upper bounds for each agent
    """
    agent_ids = env.possible_agents
    n_agents = len(agent_ids)

    obs_dims = []
    act_dims = []
    action_lows = []
    action_highs = []

    for agent_id in agent_ids:
        obs_space = env.observation_space(agent_id)
        act_space = env.action_space(agent_id)

        obs_dims.append(obs_space.shape[0])
        act_dims.append(act_space.shape[0])
        action_lows.append(act_space.low)
        action_highs.append(act_space.high)

    return n_agents, obs_dims, act_dims, action_lows, action_highs


def save_checkpoint(path: str, agent: "MADDPGAgent", step: int, best_return: float):
    """
    Save agent checkpoint.

    Args:
        path: Path to save checkpoint
        agent: MADDPG agent
        step: Current training step
        best_return: Best average return achieved
    """
    from MADDPG.maddpg.agent import MADDPGAgent

    if not isinstance(agent, MADDPGAgent):
        raise TypeError("agent must be an instance of MADDPGAgent")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = agent.state_dict()
    payload["step"] = step
    payload["best_return"] = best_return
    torch.save(payload, path)


def load_checkpoint(path: str, agent: "MADDPGAgent") -> Dict[str, Any]:
    """
    Load agent checkpoint.

    Args:
        path: Path to checkpoint file
        agent: MADDPG agent to load into

    Returns:
        data: Checkpoint data dictionary
    """
    from MADDPG.maddpg.agent import MADDPGAgent

    if not isinstance(agent, MADDPGAgent):
        raise TypeError("agent must be an instance of MADDPGAgent")

    data = torch.load(path, map_location="cpu")
    agent.load_state_dict(data)
    return data
