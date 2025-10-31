from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import yaml


def _get_device(dev: str) -> str:
    """Get device string (cuda or cpu)."""
    if dev == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return dev


@dataclass
class Config:
    """Configuration for MADDPG training."""

    # WandB
    project: str = "rl-practice"
    entity: Optional[str] = None
    run_name: str = "maddpg-run"
    wandb_key: str = ""

    # Environment
    env_id: str = "simple_spread_v3"
    render_mode: Optional[str] = None
    seed: int = 42
    n_agents: int = 3
    max_cycles: int = 25

    # Training
    total_steps: int = 1_000_000
    start_steps: int = 25_000
    batch_size: int = 1024
    buffer_size: int = 1_000_000
    updates_per_step: int = 1
    gamma: float = 0.95
    tau: float = 0.01
    actor_lr: float = 1e-2
    critic_lr: float = 1e-2
    exploration_noise: float = 0.1
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5

    # Model
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"

    # Logging & Checkpoints
    log_interval: int = 5000
    checkpoint_interval: int = 50000
    checkpoint_dir: str = "MADDPG/checkpoints"
    save_best: bool = True
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "MADDPG/logs/maddpg.log"
    log_to_console: bool = True

    # Inference
    inference_model_path: str = "MADDPG/checkpoints/best.pt"
    episodes: int = 5

    # Misc
    device: str = "auto"

    @staticmethod
    def from_yaml(path: str) -> "Config":
        """Load configuration from YAML file."""
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
        """Convert configuration to dictionary for logging."""
        return {
            "project": self.project,
            "entity": self.entity,
            "run_name": self.run_name,
            "env_id": self.env_id,
            "render_mode": self.render_mode,
            "seed": self.seed,
            "n_agents": self.n_agents,
            "max_cycles": self.max_cycles,
            "total_steps": self.total_steps,
            "start_steps": self.start_steps,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "updates_per_step": self.updates_per_step,
            "gamma": self.gamma,
            "tau": self.tau,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "exploration_noise": self.exploration_noise,
            "target_policy_noise": self.target_policy_noise,
            "target_noise_clip": self.target_noise_clip,
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
