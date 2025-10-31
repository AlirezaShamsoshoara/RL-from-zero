from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch


class MultiAgentReplayBuffer:
    """
    Replay buffer for multi-agent reinforcement learning.

    Stores transitions for all agents and samples batches for training.
    Each transition contains observations, actions, rewards, next observations,
    and done flags for all agents.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dims: List[int],
        act_dims: List[int],
        capacity: int,
        device: torch.device,
    ):
        """
        Initialize multi-agent replay buffer.

        Args:
            n_agents: Number of agents
            obs_dims: List of observation dimensions for each agent
            act_dims: List of action dimensions for each agent
            capacity: Maximum buffer size
            device: Device to use for tensor conversion
        """
        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Create separate buffers for each agent
        self.obs_bufs = [
            np.zeros((capacity, obs_dim), dtype=np.float32)
            for obs_dim in obs_dims
        ]
        self.next_obs_bufs = [
            np.zeros((capacity, obs_dim), dtype=np.float32)
            for obs_dim in obs_dims
        ]
        self.acts_bufs = [
            np.zeros((capacity, act_dim), dtype=np.float32)
            for act_dim in act_dims
        ]
        self.rews_bufs = [
            np.zeros((capacity, 1), dtype=np.float32)
            for _ in range(n_agents)
        ]
        self.done_bufs = [
            np.zeros((capacity, 1), dtype=np.float32)
            for _ in range(n_agents)
        ]

    def add(
        self,
        obs_list: List[np.ndarray],
        acts_list: List[np.ndarray],
        rewards: List[float],
        next_obs_list: List[np.ndarray],
        dones: List[bool],
    ):
        """
        Add a transition to the buffer.

        Args:
            obs_list: List of observations, one per agent
            acts_list: List of actions, one per agent
            rewards: List of rewards, one per agent
            next_obs_list: List of next observations, one per agent
            dones: List of done flags, one per agent
        """
        for i in range(self.n_agents):
            self.obs_bufs[i][self.ptr] = obs_list[i]
            self.acts_bufs[i][self.ptr] = acts_list[i]
            self.rews_bufs[i][self.ptr] = rewards[i]
            self.next_obs_bufs[i][self.ptr] = next_obs_list[i]
            self.done_bufs[i][self.ptr] = float(dones[i])

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size

    def sample(
        self, batch_size: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            obs_list: List of observation tensors, one per agent [batch_size, obs_dim]
            acts_list: List of action tensors, one per agent [batch_size, act_dim]
            rews_list: List of reward tensors, one per agent [batch_size, 1]
            next_obs_list: List of next observation tensors, one per agent [batch_size, obs_dim]
            dones_list: List of done tensors, one per agent [batch_size, 1]
        """
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs_list = []
        acts_list = []
        rews_list = []
        next_obs_list = []
        dones_list = []

        for i in range(self.n_agents):
            obs = torch.as_tensor(
                self.obs_bufs[i][idxs], device=self.device, dtype=torch.float32
            )
            acts = torch.as_tensor(
                self.acts_bufs[i][idxs], device=self.device, dtype=torch.float32
            )
            rews = torch.as_tensor(
                self.rews_bufs[i][idxs], device=self.device, dtype=torch.float32
            )
            next_obs = torch.as_tensor(
                self.next_obs_bufs[i][idxs], device=self.device, dtype=torch.float32
            )
            dones = torch.as_tensor(
                self.done_bufs[i][idxs], device=self.device, dtype=torch.float32
            )

            obs_list.append(obs)
            acts_list.append(acts)
            rews_list.append(rews)
            next_obs_list.append(next_obs)
            dones_list.append(dones)

        return obs_list, acts_list, rews_list, next_obs_list, dones_list
