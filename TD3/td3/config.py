from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

import yaml


def _get_device(device: str) -> str:
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu, cuda, or auto")
    return device


@dataclass
class Config:
    project: str = "rl-practice"
    entity: Optional[str] = None
    run_name: str = "td3-run"
    wandb_key: str = ""
    env_id: str = "MountainCarContinuous-v0"
    render_mode: Optional[str] = None
    seed: int = 42
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    total_steps: int = 200000
    start_steps: int = 10000
    update_after: Optional[int] = None
    batch_size: int = 256
    buffer_size: int = 1000000
    updates_per_step: int = 1
    num_envs: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    random_action_prob: float = 0.0
    random_action_hold_min: int = 1
    random_action_hold_max: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 0.0
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    log_interval: int = 1000
    checkpoint_interval: int = 10000
    checkpoint_dir: str = "TD3/checkpoints"
    save_best: bool = True
    best_window: int = 20
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "TD3/logs/td3.log"
    log_to_console: bool = True
    inference_model_path: str = "TD3/checkpoints/best.pt"
    episodes: int = 5
    max_steps_per_episode: int = 999
    device: str = "auto"

    def validate(self) -> "Config":
        for name in (
            "total_steps",
            "batch_size",
            "buffer_size",
            "updates_per_step",
            "num_envs",
            "policy_delay",
            "random_action_hold_min",
            "random_action_hold_max",
            "best_window",
            "episodes",
            "max_steps_per_episode",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("start_steps", "seed", "log_interval", "checkpoint_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.update_after is None:
            self.update_after = self.start_steps
        if (
            isinstance(self.update_after, bool)
            or not isinstance(self.update_after, int)
            or self.update_after < 0
        ):
            raise ValueError("update_after must be nonnegative or null")
        if self.total_steps % self.num_envs:
            raise ValueError(
                "total_steps must be divisible by num_envs for an exact vector budget"
            )
        if self.buffer_size < self.batch_size:
            raise ValueError("buffer_size must support batch_size")
        if self.random_action_hold_max < self.random_action_hold_min:
            raise ValueError("random_action_hold_max must be >= random_action_hold_min")
        for name in (
            "gamma",
            "tau",
            "target_noise",
            "noise_clip",
            "exploration_noise",
            "random_action_prob",
            "actor_lr",
            "critic_lr",
            "max_grad_norm",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("gamma", "tau"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.random_action_prob > 1:
            raise ValueError("random_action_prob must be in [0, 1]")
        for name in ("actor_lr", "critic_lr"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("save_best", "log_to_file", "log_to_console"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if not isinstance(self.hidden_sizes, (list, tuple)) or any(
            isinstance(n, bool) or not isinstance(n, int) or n <= 0
            for n in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integers or be empty")
        if self.activation not in ("relu", "tanh", "elu", "leaky_relu"):
            raise ValueError("unsupported activation")
        if not isinstance(self.env_kwargs, dict) or "render_mode" in self.env_kwargs:
            raise ValueError("env_kwargs must be a mapping without render_mode")
        if self.render_mode not in (None, "human", "rgb_array"):
            raise ValueError("unsupported render_mode")
        if not isinstance(self.env_id, str) or not self.env_id:
            raise ValueError("env_id must be nonempty")
        self.device = _get_device(self.device)
        for name in ("checkpoint_dir", "inference_model_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty path")
            setattr(self, name, os.path.normpath(value))
        if self.log_file:
            self.log_file = os.path.normpath(self.log_file)
        return self

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("configuration must be a mapping")
        allowed = {item.name for item in fields(Config)}
        return Config(
            **{key: value for key, value in data.items() if key in allowed}
        ).validate()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("wandb_key", None)
        return data
