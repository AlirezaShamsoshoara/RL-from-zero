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
    run_name: str = "mappo-run"
    # API key for programmatic login (do not log). Default blank
    wandb_key: str = ""

    # Env
    env_id: str = "multiwalker_v9"
    render_mode: Optional[str] = None
    n_walkers: int = 3
    seed: int = 42
    discretize_actions: bool = False
    action_bins: int = 3

    # Multi-Agent Settings
    n_agents: int = 3
    share_policy: bool = True  # Whether all agents share the same policy network
    use_centralized_critic: bool = True  # Use centralized critic with full state

    # Training
    total_timesteps: int = 1_000_000
    rollout_steps: int = 1024
    update_iterations: int = 10
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    anneal_lr: bool = True

    # Model
    actor_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    critic_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    activation: str = "relu"  # tanh or relu

    # Logging & Checkpoints
    log_interval: int = 10
    checkpoint_interval: int = 50
    checkpoint_dir: str = "MAPPO/checkpoints"
    save_best: bool = True
    # Python logging
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "MAPPO/logs/mappo.log"
    log_to_console: bool = True

    # Inference
    inference_model_path: str = "MAPPO/checkpoints/best.pt"
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
        if isinstance(cfg.learning_rate, str):
            try:
                cfg.learning_rate = float(cfg.learning_rate)
            except ValueError as exc:
                raise ValueError(
                    f"learning_rate must be a float, got {cfg.learning_rate!r}"
                ) from exc
        cfg.device = _get_device(cfg.device)
        # Normalize checkpoint paths to be workspace-root relative
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
            # Intentionally exclude wandb_key from the public config dict
            "env_id": self.env_id,
            "render_mode": self.render_mode,
            "n_walkers": self.n_walkers,
            "seed": self.seed,
            "discretize_actions": self.discretize_actions,
            "action_bins": self.action_bins,
            "n_agents": self.n_agents,
            "share_policy": self.share_policy,
            "use_centralized_critic": self.use_centralized_critic,
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
            "target_kl": self.target_kl,
            "anneal_lr": self.anneal_lr,
            "actor_hidden_sizes": self.actor_hidden_sizes,
            "critic_hidden_sizes": self.critic_hidden_sizes,
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
