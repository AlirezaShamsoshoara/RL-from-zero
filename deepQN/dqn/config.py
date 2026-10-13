from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch
import yaml


def _get_device(dev: str) -> str:
    return (
        "cuda"
        if dev == "auto" and torch.cuda.is_available()
        else "cpu"
        if dev == "auto"
        else dev
    )


@dataclass
class Config:
    project: str = "rl-practice"
    entity: Optional[str] = None
    run_name: str = "dqn-run"
    wandb_key: str = ""
    env_id: str = "MountainCar-v0"
    render_mode: Optional[str] = None
    seed: int = 42
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "relu"
    total_steps: int = 200_000
    learning_starts: int = 1_000
    train_freq: int = 4
    batch_size: int = 64
    buffer_size: int = 500_000
    gamma: float = 0.99
    lr: float = 3e-4
    target_update_interval: int = 1_000
    max_grad_norm: float = 10.0
    double_dqn: bool = True
    normalize_observations: bool = False
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    exploration_hold_min: int = 1
    exploration_hold_max: int = 1
    eval_epsilon: float = 0.0
    log_interval: int = 1_000
    checkpoint_interval: int = 10_000
    checkpoint_dir: str = "deepQN/checkpoints"
    save_best: bool = True
    best_window: int = 5
    max_steps_per_episode: int = 1000
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "deepQN/logs/dqn.log"
    log_to_console: bool = True
    inference_model_path: str = "deepQN/checkpoints/best.pt"
    episodes: int = 5
    device: str = "auto"

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        if not isinstance(data, dict):
            raise ValueError("configuration must be a YAML mapping")
        cfg = Config()
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        for name in (
            "gamma",
            "lr",
            "max_grad_norm",
            "epsilon_start",
            "epsilon_end",
            "eval_epsilon",
        ):
            value = float(getattr(cfg, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(cfg, name, value)
        for name in (
            "total_steps",
            "train_freq",
            "batch_size",
            "buffer_size",
            "target_update_interval",
            "log_interval",
            "checkpoint_interval",
            "best_window",
            "max_steps_per_episode",
            "exploration_hold_min",
            "exploration_hold_max",
            "episodes",
        ):
            value = getattr(cfg, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_starts", "epsilon_decay_steps", "seed"):
            value = getattr(cfg, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if cfg.seed >= 2**32:
            raise ValueError("seed must be less than 2**32")
        if not 0 <= cfg.gamma <= 1 or cfg.lr <= 0 or cfg.max_grad_norm < 0:
            raise ValueError("invalid gamma, learning rate or gradient clip")
        if (
            not 0 <= cfg.epsilon_end <= cfg.epsilon_start <= 1
            or not 0 <= cfg.eval_epsilon <= 1
        ):
            raise ValueError("invalid epsilon bounds")
        if not isinstance(cfg.double_dqn, bool) or not isinstance(
            cfg.normalize_observations, bool
        ):
            raise ValueError("double_dqn and normalize_observations must be bools")
        if cfg.exploration_hold_max < cfg.exploration_hold_min:
            raise ValueError("exploration hold bounds must be ordered")
        if cfg.buffer_size < cfg.batch_size:
            raise ValueError("buffer_size must be at least batch_size")
        if not isinstance(cfg.env_kwargs, dict):
            raise ValueError("env_kwargs must be a mapping")
        if not isinstance(cfg.hidden_sizes, list) or any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in cfg.hidden_sizes
        ):
            raise ValueError("hidden_sizes must contain positive integers")
        if cfg.activation not in {"relu", "tanh", "sigmoid", "gelu"}:
            raise ValueError("unsupported activation")
        cfg.device = _get_device(cfg.device)
        cfg.checkpoint_dir = os.path.normpath(cfg.checkpoint_dir)
        cfg.inference_model_path = os.path.normpath(cfg.inference_model_path)
        if cfg.log_file:
            cfg.log_file = os.path.normpath(cfg.log_file)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("wandb_key", None)
        return data
