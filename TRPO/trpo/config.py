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
    run_name: str = "trpo-run"
    wandb_key: str = ""

    # Environment
    env_id: str = "Pendulum-v1"
    render_mode: Optional[str] = None
    num_envs: int = 1
    seed: int = 42
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Training
    total_timesteps: int = 300_000
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_kl: float = 0.01
    cg_iters: int = 10
    cg_damping: float = 0.1
    line_search_coef: float = 0.5
    line_search_steps: int = 10
    vf_lr: float = 3e-4
    vf_iters: int = 5
    entropy_coef: float = 0.0
    normalize_advantages: bool = True

    # Model
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"

    # Logging & Checkpoints
    log_interval: int = 10
    checkpoint_interval: int = 10
    checkpoint_dir: str = "TRPO/checkpoints"
    save_best: bool = True
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "TRPO/logs/trpo.log"
    log_to_console: bool = True

    # Inference
    inference_model_path: str = "TRPO/checkpoints/best.pt"
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
            "num_envs": self.num_envs,
            "seed": self.seed,
            "env_kwargs": self.env_kwargs,
            "total_timesteps": self.total_timesteps,
            "rollout_steps": self.rollout_steps,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "max_kl": self.max_kl,
            "cg_iters": self.cg_iters,
            "cg_damping": self.cg_damping,
            "line_search_coef": self.line_search_coef,
            "line_search_steps": self.line_search_steps,
            "vf_lr": self.vf_lr,
            "vf_iters": self.vf_iters,
            "entropy_coef": self.entropy_coef,
            "normalize_advantages": self.normalize_advantages,
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

