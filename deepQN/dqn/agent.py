from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from deepQN.dqn.networks import QNetwork


@dataclass
class UpdateStats:
    loss: float
    td_error: float
    q_value: float = 0.0
    target_value: float = 0.0


class DQNAgent:
    def __init__(
        self,
        obs_space: spaces.Box,
        act_space: spaces.Discrete,
        hidden_sizes: list[int],
        activation: str,
        lr: float,
        gamma: float,
        target_update_interval: int,
        max_grad_norm: float,
        double_dqn: bool,
        device: str,
        normalize_observations: bool = False,
    ) -> None:
        if not isinstance(obs_space, spaces.Box) or len(obs_space.shape) != 1:
            raise ValueError("DQN requires flat Box observations")
        if not isinstance(act_space, spaces.Discrete) or act_space.start != 0:
            raise ValueError("DQN requires zero-based Discrete actions")
        if (
            not np.isfinite(lr)
            or lr <= 0
            or not np.isfinite(gamma)
            or not 0 <= gamma <= 1
        ):
            raise ValueError("lr must be positive and gamma in [0, 1]")
        if (
            not isinstance(target_update_interval, int)
            or isinstance(target_update_interval, bool)
            or target_update_interval <= 0
        ):
            raise ValueError("target_update_interval must be a positive integer")
        if not np.isfinite(max_grad_norm) or max_grad_norm < 0:
            raise ValueError("max_grad_norm must be finite and nonnegative")
        if not isinstance(normalize_observations, bool):
            raise ValueError("normalize_observations must be a bool")
        self.device = torch.device(device)
        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_space = act_space
        self.action_dim = int(act_space.n)
        self.gamma = float(gamma)
        self.target_update_interval = target_update_interval
        self.max_grad_norm = float(max_grad_norm)
        self.double_dqn = double_dqn
        self.normalize_observations = normalize_observations
        self._observation_low = self._observation_high = None
        self._obs_center = torch.zeros(self.obs_dim, device=self.device)
        self._obs_scale = torch.ones(self.obs_dim, device=self.device)
        if normalize_observations:
            low = torch.as_tensor(
                obs_space.low, dtype=torch.float32, device=self.device
            )
            high = torch.as_tensor(
                obs_space.high, dtype=torch.float32, device=self.device
            )
            if (
                not torch.isfinite(low).all()
                or not torch.isfinite(high).all()
                or not torch.all(high > low)
            ):
                raise ValueError(
                    "observation normalization requires finite ordered Box bounds"
                )
            self._observation_low, self._observation_high = low.clone(), high.clone()
            self._obs_center = low * 0.5 + high * 0.5
            self._obs_scale = high * 0.5 - low * 0.5
            if not torch.isfinite(self._obs_scale).all() or not torch.all(
                self._obs_scale > 0
            ):
                raise ValueError(
                    "observation normalization scale must be finite and positive"
                )
        self.q_network = QNetwork(
            self.obs_dim, self.action_dim, hidden_sizes, activation
        ).to(self.device)
        self.target_q_network = QNetwork(
            self.obs_dim, self.action_dim, hidden_sizes, activation
        ).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.step_count = 0
        self.train_steps = 0

    def _encode(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalize_observations:
            tensor = (tensor - self._obs_center) / self._obs_scale
        if not torch.isfinite(tensor).all():
            raise ValueError("encoded observations must be finite")
        return tensor

    def _observation(self, obs) -> torch.Tensor:
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(
            1, -1
        )
        if tensor.shape[1] != self.obs_dim or not torch.isfinite(tensor).all():
            raise ValueError("observation must match the finite flat observation space")
        return self._encode(tensor)

    @torch.no_grad()
    def q_value(self, obs: np.ndarray) -> float:
        return float(self.q_network(self._observation(obs)).max().item())

    def act(self, obs: np.ndarray, epsilon: float, deterministic: bool = False) -> int:
        if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be finite and in [0, 1]")
        obs_tensor = self._observation(obs)
        if not deterministic:
            self.step_count += 1
            if np.random.rand() < epsilon:
                return int(self.act_space.sample())
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def update(self, batch: Dict[str, torch.Tensor]) -> UpdateStats:
        obs = batch["obs"].to(self.device, dtype=torch.float32)
        actions_raw = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device, dtype=torch.float32)
        next_obs = batch["next_obs"].to(self.device, dtype=torch.float32)
        dones = batch["dones"].to(self.device, dtype=torch.float32)
        if obs.ndim != 2:
            raise ValueError("replay observations must be a two-dimensional batch")
        size = obs.shape[0]
        if (
            size <= 0
            or obs.shape != (size, self.obs_dim)
            or next_obs.shape != obs.shape
        ):
            raise ValueError("replay observations have invalid shapes")
        if any(t.shape != (size,) for t in (actions_raw, rewards, dones)):
            raise ValueError("replay action/reward/done shapes must match the batch")
        if any(
            not torch.isfinite(t).all()
            for t in (obs, next_obs, actions_raw, rewards, dones)
        ):
            raise ValueError("replay batch must contain finite values")
        if (
            torch.any(actions_raw != actions_raw.long())
            or torch.any(actions_raw < 0)
            or torch.any(actions_raw >= self.action_dim)
        ):
            raise ValueError("replay actions are outside the discrete action space")
        if torch.any((dones != 0) & (dones != 1)):
            raise ValueError("done masks must represent true terminations as 0 or 1")
        obs, next_obs = self._encode(obs), self._encode(next_obs)
        actions = actions_raw.long()
        action_values = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.q_network(next_obs).argmax(dim=1, keepdim=True)
                next_target_values = (
                    self.target_q_network(next_obs).gather(1, next_actions).squeeze(1)
                )
            else:
                next_target_values = self.target_q_network(next_obs).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - dones) * next_target_values
        loss = F.mse_loss(action_values, targets)
        if not torch.isfinite(loss):
            raise ValueError("DQN update produced a non-finite loss")
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.target_update_interval == 0:
            self.update_target_network()
        return UpdateStats(
            loss=float(loss.item()),
            td_error=float((action_values.detach() - targets).abs().mean().item()),
            q_value=float(action_values.detach().mean().item()),
            target_value=float(targets.mean().item()),
        )

    def update_target_network(self) -> None:
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def state_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(
            {
                "q_network": self.q_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_count": self.step_count,
                "train_steps": self.train_steps,
                "normalize_observations": self.normalize_observations,
                "observation_low": self._observation_low.cpu().tolist()
                if self.normalize_observations
                else None,
                "observation_high": self._observation_high.cpu().tolist()
                if self.normalize_observations
                else None,
            }
        )

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for name in ("step_count", "train_steps"):
            value = state.get(name, 0)
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
        saved_normalization = state.get("normalize_observations", False)
        if (
            not isinstance(saved_normalization, bool)
            or saved_normalization != self.normalize_observations
        ):
            raise ValueError(
                "checkpoint observation normalization does not match the agent"
            )
        if saved_normalization:
            for key, expected in (
                ("observation_low", self._observation_low),
                ("observation_high", self._observation_high),
            ):
                value = state.get(key)
                if value is None:
                    raise ValueError(
                        "normalized checkpoint is missing observation bounds"
                    )
                actual = torch.as_tensor(value, dtype=torch.float32, device=self.device)
                if actual.shape != expected.shape or not torch.equal(actual, expected):
                    raise ValueError(
                        "checkpoint observation bounds do not match the environment"
                    )
        self.q_network.load_state_dict(state["q_network"])
        self.target_q_network.load_state_dict(
            state.get("target_q_network", state["q_network"])
        )
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.step_count = int(state.get("step_count", 0))
        self.train_steps = int(state.get("train_steps", 0))
        # Preserve the saved target lag: no unconditional online-to-target sync.

    def to(self, device: str) -> "DQNAgent":
        self.device = torch.device(device)
        self.q_network.to(self.device)
        self.target_q_network.to(self.device)
        self._obs_center = self._obs_center.to(self.device)
        self._obs_scale = self._obs_scale.to(self.device)
        if self.normalize_observations:
            self._observation_low = self._observation_low.to(self.device)
            self._observation_high = self._observation_high.to(self.device)
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(self.device)
        return self
