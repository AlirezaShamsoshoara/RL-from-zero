from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class Config:
    project: str = "rl-practice"
    entity: Optional[str] = None
    run_name: str = "qlearning-run"
    wandb_key: str = ""
    env_id: str = "FrozenLake-v1"
    render_mode: Optional[str] = None
    seed: int = 42
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    total_episodes: int = 5000
    max_steps_per_episode: int = 200
    gamma: float = 0.99
    alpha: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.001
    log_interval: int = 50
    checkpoint_interval: int = 250
    checkpoint_dir: str = "Qlearning/checkpoints"
    save_best: bool = True
    best_window: int = 10
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "Qlearning/logs/qlearning.log"
    log_to_console: bool = True
    inference_model_path: str = "Qlearning/checkpoints/best.pt"
    episodes: int = 5
    device: str = "cpu"  # The tabular NumPy learner is CPU-only.

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
        for name in ("alpha", "gamma", "epsilon_start", "epsilon_end", "epsilon_decay"):
            value = float(getattr(cfg, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(cfg, name, value)
        for name in (
            "total_episodes",
            "max_steps_per_episode",
            "log_interval",
            "checkpoint_interval",
            "best_window",
            "episodes",
        ):
            value = getattr(cfg, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(cfg.seed, int)
            or isinstance(cfg.seed, bool)
            or not 0 <= cfg.seed < 2**32
        ):
            raise ValueError("seed must be in [0, 2**32)")
        if not 0 < cfg.alpha <= 1 or not 0 <= cfg.gamma <= 1:
            raise ValueError("alpha must be in (0, 1] and gamma in [0, 1]")
        if not 0 <= cfg.epsilon_end <= cfg.epsilon_start <= 1 or cfg.epsilon_decay < 0:
            raise ValueError("invalid epsilon bounds or decay")
        if not isinstance(cfg.env_kwargs, dict):
            raise ValueError("env_kwargs must be a mapping")
        if cfg.device not in {"cpu", "auto"}:
            raise ValueError(
                "tabular Q-learning supports only CPU (device cpu or auto)"
            )
        cfg.device = "cpu"
        cfg.checkpoint_dir = os.path.normpath(cfg.checkpoint_dir)
        cfg.inference_model_path = os.path.normpath(cfg.inference_model_path)
        if cfg.log_file:
            cfg.log_file = os.path.normpath(cfg.log_file)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("wandb_key", None)
        return data
