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
    # Blend between local and global reward on MPE cooperative tasks.
    # 0.0 = purely global (MADDPG paper setup); 0.5 = mixed.
    local_ratio: float = 0.0

    # Training
    total_steps: int = 1_000_000
    start_steps: int = 25_000
    batch_size: int = 1024
    buffer_size: int = 1_000_000
    # Gradient-update cadence: perform `updates_per_step` gradient updates every
    # `update_every` env steps. The original MADDPG paper does 1 update per
    # episode (~25 env steps); running 1 update per env step is 25x more and
    # destabilizes the joint-action critic on cooperative MPE tasks.
    updates_per_step: int = 1
    update_every: int = 25
    gamma: float = 0.95
    tau: float = 0.01
    actor_lr: float = 1e-2
    critic_lr: float = 1e-2
    exploration_noise: float = 0.1        # starting Gaussian noise std at collect time
    exploration_noise_end: float = 0.05   # final noise std after decay_steps
    exploration_decay_steps: int = 100_000  # linear decay from start to end over this many env steps
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    grad_clip: float = 0.5  # global-norm clip on actor + critic; <=0 disables

    # Model
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    # Optional separate size for the centralized critic; falls back to
    # `hidden_sizes` when None. The joint-action critic input space is much
    # larger than a single agent's observation (obs*N + act*N), so it often
    # benefits from more capacity than the decentralized actor.
    critic_hidden_sizes: Optional[list[int]] = None
    activation: str = "relu"
    layer_norm: bool = False              # LayerNorm after each hidden Linear
    twin_critics: bool = False            # MATD3-style min-of-two critics on the target
    policy_delay: int = 1                 # actor + target updates every N critic updates

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
            "local_ratio": self.local_ratio,
            "total_steps": self.total_steps,
            "start_steps": self.start_steps,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "updates_per_step": self.updates_per_step,
            "update_every": self.update_every,
            "gamma": self.gamma,
            "tau": self.tau,
            "actor_lr": self.actor_lr,
            "critic_lr": self.critic_lr,
            "exploration_noise": self.exploration_noise,
            "exploration_noise_end": self.exploration_noise_end,
            "exploration_decay_steps": self.exploration_decay_steps,
            "target_policy_noise": self.target_policy_noise,
            "target_noise_clip": self.target_noise_clip,
            "grad_clip": self.grad_clip,
            "hidden_sizes": self.hidden_sizes,
            "critic_hidden_sizes": self.critic_hidden_sizes,
            "activation": self.activation,
            "layer_norm": self.layer_norm,
            "twin_critics": self.twin_critics,
            "policy_delay": self.policy_delay,
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
