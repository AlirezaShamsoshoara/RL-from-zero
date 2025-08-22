from __future__ import annotations
import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict


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
    run_name: str = "ppo-run"

    # Env
    env_id: str = "CartPole-v1"
    render_mode: Optional[str] = None
    num_envs: int = 1
    seed: int = 42

    # Training
    total_timesteps: int = 100_000
    rollout_steps: int = 2048
    update_iterations: int = 4
    minibatch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Model
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"  # tanh or relu

    # Logging & Checkpoints
    log_interval: int = 10
    checkpoint_interval: int = 10
    checkpoint_dir: str = "PPO/checkpoints"
    save_best: bool = True

    # Inference
    inference_model_path: str = "PPO/checkpoints/best.pt"
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
        # Normalize checkpoint paths to be workspace-root relative
        cfg.checkpoint_dir = os.path.normpath(cfg.checkpoint_dir)
        cfg.inference_model_path = os.path.normpath(cfg.inference_model_path)
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
            "total_timesteps": self.total_timesteps,
            "rollout_steps": self.rollout_steps,
            "update_iterations": self.update_iterations,
            "minibatch_size": self.minibatch_size,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_coef": self.clip_coef,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "log_interval": self.log_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "save_best": self.save_best,
            "inference_model_path": self.inference_model_path,
            "episodes": self.episodes,
            "device": self.device,
        }
