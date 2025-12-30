from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import yaml


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
    run_name: str = "dqn-run"
    wandb_key: str = ""

    # Environment
    env_id: str = "MountainCar-v0"
    render_mode: Optional[str] = None
    seed: int = 42
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Model
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    activation: str = "relu"

    # Training
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

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    eval_epsilon: float = 0.0

    # Logging & Checkpoints
    log_interval: int = 1_000
    checkpoint_interval: int = 10_000
    checkpoint_dir: str = "deepQN/checkpoints"
    save_best: bool = True
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "deepQN/logs/dqn.log"
    log_to_console: bool = True

    # Inference
    inference_model_path: str = "deepQN/checkpoints/best.pt"
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
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "total_steps": self.total_steps,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "lr": self.lr,
            "target_update_interval": self.target_update_interval,
            "max_grad_norm": self.max_grad_norm,
            "double_dqn": self.double_dqn,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "eval_epsilon": self.eval_epsilon,
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
