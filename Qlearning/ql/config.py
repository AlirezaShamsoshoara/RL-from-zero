from __future__ import annotations
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Any, Dict


def _get_device(dev: str) -> str:
    if dev == "auto":
        try:
            import torch  # noqa: F401
            import torch.cuda as _cuda
            return "cuda" if _cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return dev


@dataclass
class Config:
    # WandB
    project: str = "rl-practice"
    entity: Optional[str] = None
    run_name: str = "qlearning-run"
    wandb_key: str = ""  # optional key for programmatic login

    # Environment
    env_id: str = "FrozenLake-v1"
    render_mode: Optional[str] = None
    seed: int = 42
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Training
    total_episodes: int = 5000
    max_steps_per_episode: int = 200
    gamma: float = 0.99
    alpha: float = 0.1  # learning rate for Q updates
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.001  # exponential decay per step

    # Logging & Checkpoints
    log_interval: int = 50
    checkpoint_interval: int = 250
    checkpoint_dir: str = "Qlearning/checkpoints"
    save_best: bool = True
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "Qlearning/logs/qlearning.log"
    log_to_console: bool = True

    # Inference
    inference_model_path: str = "Qlearning/checkpoints/best.pt"
    episodes: int = 5

    # Misc
    device: str = "auto"

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        cfg = Config()
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.device = _get_device(cfg.device)
        cfg.checkpoint_dir = os.path.normpath(cfg.checkpoint_dir)
        cfg.inference_model_path = os.path.normpath(cfg.inference_model_path)
        if cfg.log_file:
            cfg.log_file = os.path.normpath(cfg.log_file)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": self.project,
            "entity": self.entity,
            "run_name": self.run_name,
            "env_id": self.env_id,
            "render_mode": self.render_mode,
            "seed": self.seed,
            "env_kwargs": self.env_kwargs,
            "total_episodes": self.total_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "log_interval": self.log_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "save_best": self.save_best,
            "log_level": self.log_level,
            "log_to_file": self.log_to_file,
            "log_file": self.log_file,
            "log_to_console": self.log_to_console,
            "inference_model_path": self.inference_model_path,
            "episodes": self.episodes,
            "device": self.device,
        }

