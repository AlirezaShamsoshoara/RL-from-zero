from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

import yaml


def _get_device(dev: str) -> str:
    if dev not in ("auto", "cpu"):
        raise ValueError("A3C shared-memory training supports only cpu (or auto)")
    return "cpu"


@dataclass
class Config:
    project: str = "rl-practice"
    entity: Optional[str] = None
    run_name: str = "a3c-lunarlander"
    wandb_key: str = ""
    env_id: str = "LunarLander-v3"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    render_mode: Optional[str] = None
    seed: int = 42
    num_workers: int = 4
    num_envs: int = 1
    t_max: int = 20
    total_steps: int = 1_000_000
    gamma: float = 0.99
    entropy_coef: float = 0.05
    value_loss_coef: float = 0.01
    learning_rate: float = 1e-3
    max_grad_norm: float = 40.0
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    log_interval: int = 1000
    checkpoint_interval: int = 10000
    checkpoint_dir: str = "A3C/checkpoints"
    save_best: bool = True
    best_window: int = 20
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "A3C/logs/a3c.log"
    log_to_console: bool = True
    inference_model_path: str = "A3C/checkpoints/best.pt"
    episodes: int = 5
    max_steps_per_episode: int = 1000
    anneal_lr: bool = True
    anneal_entropy: bool = False
    entropy_coef_end: float = 0.01
    normalize_advantages: bool = False
    reward_scale: float = 1.0
    reward_shaping: bool = False
    worker_timeout: float = 120.0
    shutdown_timeout: float = 10.0
    device: str = "cpu"

    def validate(self) -> "Config":
        for name in (
            "num_workers",
            "num_envs",
            "t_max",
            "total_steps",
            "episodes",
            "best_window",
            "max_steps_per_episode",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("log_interval", "checkpoint_interval", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.total_steps % self.num_envs:
            raise ValueError(
                "total_steps must be divisible by num_envs for exact batched budgets"
            )
        for name in (
            "save_best",
            "log_to_file",
            "log_to_console",
            "anneal_lr",
            "anneal_entropy",
            "normalize_advantages",
            "reward_shaping",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "gamma",
            "entropy_coef",
            "value_loss_coef",
            "learning_rate",
            "max_grad_norm",
            "entropy_coef_end",
            "reward_scale",
            "worker_timeout",
            "shutdown_timeout",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        for name in (
            "learning_rate",
            "reward_scale",
            "worker_timeout",
            "shutdown_timeout",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.hidden_sizes, (list, tuple)) or not self.hidden_sizes:
            raise ValueError("hidden_sizes must be a nonempty sequence")
        if any(
            isinstance(n, bool) or not isinstance(n, int) or n <= 0
            for n in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes entries must be positive integers")
        if self.activation not in ("relu", "tanh", "elu", "leaky_relu"):
            raise ValueError("unsupported activation")
        if not isinstance(self.env_kwargs, dict):
            raise ValueError("env_kwargs must be a mapping")
        if "render_mode" in self.env_kwargs:
            raise ValueError("render_mode belongs in the top-level config")
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
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("configuration must be a mapping")
        allowed = {item.name for item in fields(Config)}
        cfg = Config(**{key: value for key, value in data.items() if key in allowed})
        return cfg.validate()

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.pop("wandb_key", None)
        return result
