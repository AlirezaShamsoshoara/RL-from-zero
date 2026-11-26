from __future__ import annotations

from numbers import Integral

import numpy as np
import torch


def _positive_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _copy_float32(value, name: str, binary: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real values")
    if binary and not np.logical_or(raw == 0, raw == 1).all():
        raise ValueError(f"{name} must contain only 0 or 1")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must remain finite after float32 conversion")
    return result


class ReplayBuffer:
    """Owned float32 ring storage with legacy global NumPy replacement sampling."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: torch.device):
        self.obs_dim = _positive_count(obs_dim, "obs_dim")
        self.act_dim = _positive_count(act_dim, "act_dim")
        self.capacity = _positive_count(capacity, "capacity")
        self.device = torch.device(device)
        self.obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros_like(self.obs_buf)
        self.acts_buf = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros_like(self.rews_buf)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, reward, next_obs, done):
        # Copy and validate every field before changing any storage or counters.
        obs = _copy_float32(obs, "obs")
        act = _copy_float32(act, "act")
        reward = _copy_float32(reward, "reward")
        next_obs = _copy_float32(next_obs, "next_obs")
        done = _copy_float32(done, "done", binary=True)
        if obs.shape != (self.obs_dim,) or next_obs.shape != (self.obs_dim,):
            raise ValueError(f"obs and next_obs must have shape ({self.obs_dim},)")
        if act.shape != (self.act_dim,):
            raise ValueError(f"act must have shape ({self.act_dim},)")
        if reward.shape not in ((), (1,)) or done.shape not in ((), (1,)):
            raise ValueError("reward and done must be scalars or shape (1,)")
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = reward.reshape(1)
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done.reshape(1)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        batch_size = _positive_count(batch_size, "batch_size")
        return self.size >= batch_size

    def add_batch(self, obs, act, reward, next_obs, done):
        obs = _copy_float32(obs, "obs")
        act = _copy_float32(act, "act")
        reward = _copy_float32(reward, "reward")
        next_obs = _copy_float32(next_obs, "next_obs")
        done = _copy_float32(done, "done", binary=True)
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim or obs.shape[0] == 0:
            raise ValueError(f"obs must have nonempty shape (N, {self.obs_dim})")
        batch_size = obs.shape[0]
        if next_obs.shape != obs.shape or act.shape != (batch_size, self.act_dim):
            raise ValueError("next_obs and act must match the observation batch")
        if reward.shape not in ((batch_size,), (batch_size, 1)):
            raise ValueError("reward must have shape (N,) or (N, 1)")
        if done.shape not in ((batch_size,), (batch_size, 1)):
            raise ValueError("done must have shape (N,) or (N, 1)")
        reward = reward.reshape(batch_size, 1)
        done = done.reshape(batch_size, 1)

        # Keep only the final capacity rows, but account for every skipped row
        # in the destination index. This is exactly sequential add, even after
        # multiple wraps. All input rows, including skipped rows, were validated.
        skip = max(0, batch_size - self.capacity)
        indices = (self.ptr + skip + np.arange(batch_size - skip)) % self.capacity
        for buffer, values in (
            (self.obs_buf, obs),
            (self.acts_buf, act),
            (self.rews_buf, reward),
            (self.next_obs_buf, next_obs),
            (self.done_buf, done),
        ):
            buffer[indices] = values[skip:]
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int):
        """Sample with replacement; any positive count is valid if nonempty.

        can_sample retains the older size >= count warmup predicate. Sampling
        itself may request more than size, since indices are drawn with replacement.
        """
        batch_size = _positive_count(batch_size, "batch_size")
        if self.size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        idxs = np.random.randint(0, self.size, size=batch_size)
        # Advanced indexing owns its array, so CPU tensors cannot alias storage.
        return tuple(
            torch.as_tensor(buffer[idxs], device=self.device, dtype=torch.float32)
            for buffer in (
                self.obs_buf,
                self.acts_buf,
                self.rews_buf,
                self.next_obs_buf,
                self.done_buf,
            )
        )
