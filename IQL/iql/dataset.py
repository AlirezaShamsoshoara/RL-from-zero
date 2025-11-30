from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import logging
import numpy as np
import torch
import gymnasium as gym


@dataclass
class DatasetStats:
    size: int
    obs_dim: int
    act_dim: int
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    terminal_fraction: float


class OfflineDataset:
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        device: torch.device,
    ):
        self.device = device
        self.size = int(observations.shape[0])
        self.obs_dim = int(observations.shape[1])
        self.act_dim = int(actions.shape[1])

        self._reward_mean = float(np.mean(rewards))
        self._reward_std = float(np.std(rewards))
        self._reward_min = float(np.min(rewards))
        self._reward_max = float(np.max(rewards))
        self._terminal_fraction = float(np.mean(dones))

        self._observations = torch.as_tensor(observations, dtype=torch.float32)
        self._actions = torch.as_tensor(actions, dtype=torch.float32)
        self._rewards = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1, 1)
        self._next_observations = torch.as_tensor(
            next_observations, dtype=torch.float32
        )
        self._dones = torch.as_tensor(dones, dtype=torch.float32).reshape(-1, 1)

    @classmethod
    def from_arrays(
        cls,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        dones: np.ndarray,
        device: torch.device,
    ) -> "OfflineDataset":
        return cls(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            device=device,
        )

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = self._observations[idx].to(self.device)
        actions = self._actions[idx].to(self.device)
        rewards = self._rewards[idx].to(self.device)
        next_obs = self._next_observations[idx].to(self.device)
        dones = self._dones[idx].to(self.device)
        return obs, actions, rewards, next_obs, dones

    @property
    def stats(self) -> DatasetStats:
        return DatasetStats(
            size=self.size,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            reward_mean=self._reward_mean,
            reward_std=self._reward_std,
            reward_min=self._reward_min,
            reward_max=self._reward_max,
            terminal_fraction=self._terminal_fraction,
        )


def build_dataset(
    source: str,
    env_id: str,
    seed: int,
    device: torch.device,
    num_steps: int,
    path: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    reward_scale: float = 1.0,
    reward_shift: float = 0.0,
    normalize_rewards: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[OfflineDataset, DatasetStats]:
    env_kwargs = env_kwargs or {}
    src = source.lower()
    if src == "random":
        arrays = _collect_random_dataset(env_id, num_steps, seed, env_kwargs)
    elif src == "npz":
        if not path:
            raise ValueError("dataset_path must be provided when dataset_source='npz'")
        arrays = _load_npz_dataset(path)
    elif src == "d4rl":
        arrays = _load_d4rl_dataset(env_id, env_kwargs)
    else:
        raise ValueError(f"Unsupported dataset_source: {source}")

    observations, actions, rewards, next_observations, dones = arrays

    rewards = rewards.astype(np.float32)
    if normalize_rewards:
        mean = float(rewards.mean())
        std = float(rewards.std() + 1e-6)
        rewards = (rewards - mean) / std
    if reward_scale != 1.0:
        rewards = rewards * reward_scale
    if reward_shift != 0.0:
        rewards = rewards + reward_shift

    dataset = OfflineDataset.from_arrays(
        observations=observations.astype(np.float32),
        actions=actions.astype(np.float32),
        rewards=rewards,
        next_observations=next_observations.astype(np.float32),
        dones=dones.astype(np.float32),
        device=device,
    )
    stats = dataset.stats
    if logger:
        logger.info(
            "Loaded dataset: size=%d | obs_dim=%d | act_dim=%d | reward_mean=%.3f ± %.3f",
            stats.size,
            stats.obs_dim,
            stats.act_dim,
            stats.reward_mean,
            stats.reward_std,
        )
    return dataset, stats


def _collect_random_dataset(
    env_id: str,
    num_steps: int,
    seed: int,
    env_kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive when using random dataset source")
    env = gym.make(env_id, **env_kwargs)
    obs, _ = env.reset(seed=seed)

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for step in range(num_steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(np.asarray(obs, dtype=np.float32))
        actions.append(np.asarray(action, dtype=np.float32))
        rewards.append(float(reward))
        next_observations.append(np.asarray(next_obs, dtype=np.float32))
        dones.append(1.0 if terminated else 0.0)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    return (
        np.stack(observations),
        np.stack(actions),
        np.asarray(rewards, dtype=np.float32),
        np.stack(next_observations),
        np.asarray(dones, dtype=np.float32),
    )


def _load_npz_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    observations = data["observations"]
    actions = data["actions"]
    rewards = data["rewards"]
    next_observations = data["next_observations"]
    if "terminals" in data:
        dones = data["terminals"]
    elif "dones" in data:
        dones = data["dones"]
    else:
        raise KeyError("Dataset NPZ must contain 'terminals' or 'dones'")
    if "timeouts" in data:
        dones = np.maximum(dones, data["timeouts"])
    return observations, actions, rewards, next_observations, dones


def _load_d4rl_dataset(
    env_id: str,
    env_kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        import d4rl  # type: ignore  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "d4rl package is required to load datasets with dataset_source='d4rl'"
        ) from exc

    env = gym.make(env_id, **env_kwargs)
    dataset = env.get_dataset()
    env.close()

    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    next_observations = dataset["next_observations"]
    terminals = dataset["terminals"]
    timeouts = dataset.get("timeouts")
    if timeouts is not None:
        dones = np.maximum(terminals, timeouts)
    else:
        dones = terminals
    return observations, actions, rewards, next_observations, dones
