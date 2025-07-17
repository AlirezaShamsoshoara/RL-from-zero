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
        except Exception:  # pragma: no cover - defensive
            return "cpu"
    return dev


@dataclass
class Config:
    # W&B
    project: str = "rl-practice"
    entity: Optional[str] = None
    run_name: str = "iql-run"
    wandb_key: str = ""

    # Environment
    env_id: str = "Pendulum-v1"
    render_mode: Optional[str] = None
    seed: int = 42
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Dataset
    dataset_source: str = "random"  # random | npz | d4rl
    dataset_env_id: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_env_kwargs: Dict[str, Any] = field(default_factory=dict)
    dataset_steps: int = 50_000
    reward_scale: float = 1.0
    reward_shift: float = 0.0
    normalize_rewards: bool = False

    # Training
    total_updates: int = 200_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7
    temperature: float = 3.0
    max_weight: float = 100.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"

    # Logging & checkpoints
    log_interval: int = 1000
    eval_interval: int = 10_000
    checkpoint_interval: int = 20_000
    checkpoint_dir: str = "IQL/checkpoints"
    save_best: bool = True
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "IQL/logs/iql.log"
    log_to_console: bool = True

    # Evaluation / inference
    eval_episodes: int = 5
    inference_model_path: str = "IQL/checkpoints/best.pt"
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
        cfg.dataset_source = str(cfg.dataset_source).lower()
        cfg.checkpoint_dir = os.path.normpath(cfg.checkpoint_dir)
        cfg.inference_model_path = os.path.normpath(cfg.inference_model_path)
        if cfg.dataset_path:
            cfg.dataset_path = os.path.normpath(cfg.dataset_path)
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
            "dataset_source": self.dataset_source,
            "dataset_env_id": self.dataset_env_id,
            "dataset_path": self.dataset_path,
            "dataset_env_kwargs": self.dataset_env_kwargs,
            "dataset_steps": self.dataset_steps,
            "reward_scale": self.reward_scale,
            "reward_shift": self.reward_shift,
            "normalize_rewards": self.normalize_rewards,
            "total_updates": self.total_updates,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "tau": self.tau,
            "expectile": self.expectile,
            "temperature": self.temperature,
            "max_weight": self.max_weight,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "value_lr": self.value_lr,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "log_interval": self.log_interval,
            "eval_interval": self.eval_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "save_best": self.save_best,
            "log_level": self.log_level,
            "log_to_file": self.log_to_file,
            "log_file": self.log_file,
            "log_to_console": self.log_to_console,
            "eval_episodes": self.eval_episodes,
            "inference_model_path": self.inference_model_path,
            "episodes": self.episodes,
            "device": self.device,
        }
