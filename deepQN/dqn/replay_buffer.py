from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, capacity: int, device: torch.device) -> None:
        for name, value in (("obs_dim", obs_dim), ("capacity", capacity)):
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        self.obs_dim = int(obs_dim)
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = self.size = 0

    def _observation(self, value) -> np.ndarray:
        array = np.asarray(value).reshape(-1)
        if (
            array.size != self.obs_dim
            or not np.all(np.isfinite(array))
            or np.any(np.abs(array) > float(np.finfo(np.float32).max))
        ):
            raise ValueError(
                "replay observation must match the finite observation dimension"
            )
        return array.astype(np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        obs_array = self._observation(obs)
        next_array = self._observation(next_obs)
        if (
            not isinstance(action, (int, np.integer))
            or isinstance(action, bool)
            or action < 0
        ):
            raise ValueError("replay action must be a nonnegative integer")
        if not np.isfinite(reward) or abs(float(reward)) > float(
            np.finfo(np.float32).max
        ):
            raise ValueError("replay reward must be finite")
        if done not in (0, 1):
            raise ValueError("replay done mask must be 0 or 1")
        self.obs[self.ptr] = obs_array
        self.next_obs[self.ptr] = next_array
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        if (
            not isinstance(batch_size, (int, np.integer))
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        return self.size >= batch_size

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if not self.can_sample(batch_size):
            raise ValueError("Insufficient samples in buffer")
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[indices], device=self.device),
            "actions": torch.as_tensor(self.actions[indices], device=self.device),
            "rewards": torch.as_tensor(self.rewards[indices], device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[indices], device=self.device),
            "dones": torch.as_tensor(self.dones[indices], device=self.device),
        }

    def __len__(self) -> int:
        return self.size
