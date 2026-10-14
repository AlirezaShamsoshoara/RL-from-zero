from __future__ import annotations

from numbers import Integral

import numpy as np
import torch


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _float32_value(value, shape: tuple[int, ...], name: str, allow_bool=False):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite real value with shape {shape}"
        ) from exc
    kinds = "biuf" if allow_bool else "iuf"
    if array.shape != shape or array.dtype.kind not in kinds:
        raise ValueError(f"{name} must be a real value with shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    limit = np.finfo(np.float32).max
    if np.any(array > limit) or np.any(array < -limit):
        raise ValueError(f"{name} must be representable as finite float32")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(array, dtype=np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be representable as finite float32")
    return result


class ReplayBuffer:
    """Owned float32 transition storage with global NumPy RNG compatibility.

    Sampling is with replacement. can_sample is the historical training gate
    (size >= batch_size), while sample itself only needs a nonempty buffer.
    done records true terminal transitions only, not time-limit truncations.
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: torch.device):
        self.obs_dim = _positive_integer(obs_dim, "obs_dim")
        self.act_dim = _positive_integer(act_dim, "act_dim")
        self.capacity = _positive_integer(capacity, "capacity")
        self.device = torch.device(device)
        self.obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, reward, next_obs, done):
        # Validate and copy every component before writing any ring slot.
        obs_value = _float32_value(obs, (self.obs_dim,), "obs")
        act_value = _float32_value(act, (self.act_dim,), "act")
        reward_value = _float32_value(reward, (), "reward")
        next_obs_value = _float32_value(next_obs, (self.obs_dim,), "next_obs")
        done_value = _float32_value(done, (), "done", allow_bool=True)
        # Check the original value, so float32 rounding cannot accept a soft mask.
        if np.asarray(done).item() not in (0, 1):
            raise ValueError("done must be exactly 0 or 1 (bool is allowed)")
        self.obs_buf[self.ptr] = obs_value
        self.acts_buf[self.ptr] = act_value
        self.rews_buf[self.ptr, 0] = reward_value.item()
        self.next_obs_buf[self.ptr] = next_obs_value
        self.done_buf[self.ptr, 0] = done_value.item()
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        batch_size = _positive_integer(batch_size, "batch_size")
        return self.size >= batch_size

    def sample(self, batch_size: int):
        batch_size = _positive_integer(batch_size, "batch_size")
        if self.size == 0:
            raise ValueError("Cannot sample an empty replay buffer")
        idxs = np.random.randint(0, self.size, size=batch_size)
        # Advanced indexing owns a copy; tensor() also keeps CPU samples from
        # aliasing either the buffer or another returned batch.
        return tuple(
            torch.tensor(buffer[idxs], device=self.device, dtype=torch.float32)
            for buffer in (
                self.obs_buf,
                self.acts_buf,
                self.rews_buf,
                self.next_obs_buf,
                self.done_buf,
            )
        )
