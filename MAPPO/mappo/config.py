from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


@dataclass
class Config:
    project: str = "rl-practice"
    entity: Optional[str] = None
    run_name: str = "mappo-run"
    wandb_key: str = ""
    env_id: str = "multiwalker_v9"
    render_mode: Optional[str] = None
    n_walkers: int = 3
    seed: int = 42
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    discretize_actions: bool = False
    action_bins: int = 3
    n_agents: int = 3
    share_policy: bool = True
    use_centralized_critic: bool = True
    total_timesteps: int = 1_000_000
    rollout_steps: int = 1024
    update_iterations: int = 10
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    critic_learning_rate: Optional[float] = None
    separate_grad_clipping: bool = False
    clip_value_loss: bool = True
    normalize_inputs: bool = False
    num_envs: int = 1
    init_checkpoint: Optional[str] = None
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    anneal_lr: bool = True
    actor_hidden_sizes: list[int] = field(default_factory=lambda: [256, 128])
    critic_hidden_sizes: list[int] = field(default_factory=lambda: [256, 128])
    activation: str = "relu"
    log_interval: int = 10
    checkpoint_interval: int = 50
    checkpoint_dir: str = "MAPPO/checkpoints"
    save_best: bool = True
    best_window: int = 10
    max_steps_per_episode: int = 500
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: str = "MAPPO/logs/mappo.log"
    log_to_console: bool = True
    inference_model_path: str = "MAPPO/checkpoints/best.pt"
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
        for key in (
            "n_agents",
            "num_envs",
            "n_walkers",
            "total_timesteps",
            "rollout_steps",
            "update_iterations",
            "minibatch_size",
            "log_interval",
            "checkpoint_interval",
            "best_window",
            "max_steps_per_episode",
            "episodes",
            "action_bins",
        ):
            value = getattr(cfg, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if cfg.n_agents != cfg.n_walkers:
            raise ValueError("n_agents must equal n_walkers")
        if cfg.action_bins < 2:
            raise ValueError("action_bins must be at least two")
        if (
            not isinstance(cfg.seed, int)
            or isinstance(cfg.seed, bool)
            or not 0 <= cfg.seed < 2**32
        ):
            raise ValueError("seed must be an unsigned 32-bit integer")
        for key in (
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_coef",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "target_kl",
        ):
            value = float(getattr(cfg, key))
            if not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
            setattr(cfg, key, value)
        if (
            cfg.learning_rate <= 0
            or not 0 <= cfg.gamma <= 1
            or not 0 <= cfg.gae_lambda <= 1
            or not 0 < cfg.clip_coef < 1
        ):
            raise ValueError("invalid learning rate, gamma, lambda or clipping range")
        if any(
            getattr(cfg, k) < 0
            for k in ("ent_coef", "vf_coef", "max_grad_norm", "target_kl")
        ):
            raise ValueError(
                "loss weights, gradient clip and target KL must be nonnegative"
            )
        if cfg.critic_learning_rate is not None:
            cfg.critic_learning_rate = float(cfg.critic_learning_rate)
            if (
                not np.isfinite(cfg.critic_learning_rate)
                or cfg.critic_learning_rate <= 0
            ):
                raise ValueError("critic_learning_rate must be positive")
        for key in (
            "separate_grad_clipping",
            "clip_value_loss",
            "normalize_inputs",
            "share_policy",
            "use_centralized_critic",
            "discretize_actions",
            "anneal_lr",
            "save_best",
        ):
            if not isinstance(getattr(cfg, key), bool):
                raise ValueError(f"{key} must be a bool")
        for key in ("actor_hidden_sizes", "critic_hidden_sizes"):
            if not isinstance(getattr(cfg, key), list) or any(
                not isinstance(v, int) or isinstance(v, bool) or v <= 0
                for v in getattr(cfg, key)
            ):
                raise ValueError(f"{key} must contain positive integers")
        if cfg.activation not in {"relu", "tanh"}:
            raise ValueError("unsupported activation")
        if cfg.env_id != "multiwalker_v9" or not isinstance(cfg.env_kwargs, dict):
            raise ValueError(
                "MAPPO currently supports multiwalker_v9 with mapping env_kwargs"
            )
        if "max_cycles" in cfg.env_kwargs:
            value = cfg.env_kwargs["max_cycles"]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("env_kwargs.max_cycles must be a positive integer")
        if "n_walkers" in cfg.env_kwargs or "render_mode" in cfg.env_kwargs:
            raise ValueError(
                "walker count and rendering have explicit configuration fields"
            )
        if cfg.device == "auto":
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        if cfg.init_checkpoint is not None:
            if not isinstance(cfg.init_checkpoint, str) or not cfg.init_checkpoint:
                raise ValueError("init_checkpoint must be a nonempty path or null")
            cfg.init_checkpoint = os.path.normpath(cfg.init_checkpoint)
        cfg.checkpoint_dir = os.path.normpath(cfg.checkpoint_dir)
        cfg.inference_model_path = os.path.normpath(cfg.inference_model_path)
        if cfg.log_file:
            cfg.log_file = os.path.normpath(cfg.log_file)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("wandb_key", None)
        return data
