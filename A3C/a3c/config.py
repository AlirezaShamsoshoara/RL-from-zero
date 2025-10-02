from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import yaml


def _get_device(dev: str) -> str:
    if dev == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return dev


@dataclass
class Config:
    # WandB
    project: str = "rl-practice"
    entity: Optional[str] = None
    run_name: str = "a3c-run"
    wandb_key: str = ""

    # Environment
    env_id: str = "CartPole-v1"
    render_mode: Optional[str] = None
    seed: int = 42

    # Workers
    num_workers: int = 4
    t_max: int = 20

    # Training
    total_steps: int = 500_000
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    learning_rate: float = 1e-4
    max_grad_norm: float = 40.0

    # Model
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    activation: str = "relu"

    # Logging & Checkpoints
    log_interval: int = 1000
    checkpoint_interval: int = 10000
    checkpoint_dir: str = "A3C/checkpoints"
    save_best: bool = True
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "A3C/logs/a3c.log"
    log_to_console: bool = True

    # Inference
    inference_model_path: str = "A3C/checkpoints/best.pt"
    episodes: int = 5

    # Misc
    device: str = "auto"

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}
        cfg = Config()
        for key, value in data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
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
            "num_workers": self.num_workers,
            "t_max": self.t_max,
            "total_steps": self.total_steps,
            "gamma": self.gamma,
            "entropy_coef": self.entropy_coef,
            "value_loss_coef": self.value_loss_coef,
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
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
