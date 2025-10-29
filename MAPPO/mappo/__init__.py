"""
Multi-Agent Proximal Policy Optimization (MAPPO)

A multi-agent reinforcement learning algorithm that extends PPO to cooperative
multi-agent settings with centralized training and decentralized execution.
"""

from .config import Config
from .agent import MAPPOAgent, Batch
from .networks import Actor, CentralizedCritic, DecentralizedCritic, ActorCritic
from .utils import set_seed, make_multiwalker_env, get_space_dims, save_checkpoint, load_checkpoint
from .logging_utils import setup_logger

__all__ = [
    "Config",
    "MAPPOAgent",
    "Batch",
    "Actor",
    "CentralizedCritic",
    "DecentralizedCritic",
    "ActorCritic",
    "set_seed",
    "make_multiwalker_env",
    "get_space_dims",
    "save_checkpoint",
    "load_checkpoint",
    "setup_logger",
]
