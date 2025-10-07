from __future__ import annotations
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: torch.device):
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs_buf[idxs], device=self.device, dtype=torch.float32)
        acts = torch.as_tensor(self.acts_buf[idxs], device=self.device, dtype=torch.float32)
        rews = torch.as_tensor(self.rews_buf[idxs], device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(self.next_obs_buf[idxs], device=self.device, dtype=torch.float32)
        done = torch.as_tensor(self.done_buf[idxs], device=self.device, dtype=torch.float32)
        return obs, acts, rews, next_obs, done
