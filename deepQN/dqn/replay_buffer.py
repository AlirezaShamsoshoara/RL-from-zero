from __future__ import annotations
from typing import Dict
import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        obs_dim: int,
        capacity: int,
        device: torch.device,
    ) -> None:
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.next_obs[self.ptr] = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if not self.can_sample(batch_size):
            raise ValueError("Insufficient samples in buffer")
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "obs": torch.as_tensor(self.obs[idxs], device=self.device),
            "actions": torch.as_tensor(self.actions[idxs], device=self.device),
            "rewards": torch.as_tensor(self.rewards[idxs], device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[idxs], device=self.device),
            "dones": torch.as_tensor(self.dones[idxs], device=self.device),
        }
        return batch

    def __len__(self) -> int:
        return self.size
