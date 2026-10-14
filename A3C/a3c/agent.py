from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch.nn.utils import clip_grad_norm_

from .networks import ActorCritic
from .utils import SharedAdam


@dataclass
class A3CStats:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    value_mean: float = 0.0
    return_mean: float = 0.0
    advantage_mean: float = 0.0


class A3CAgent:
    def __init__(
        self,
        obs_space: spaces.Box,
        act_space: spaces.Discrete,
        hidden_sizes: Iterable[int],
        activation: str,
        learning_rate: float,
        entropy_coef: float,
        value_loss_coef: float,
        max_grad_norm: float,
        device: str = "cpu",
        update_lock=None,
    ) -> None:
        if not isinstance(obs_space, spaces.Box) or len(obs_space.shape) != 1:
            raise ValueError("A3C requires flat Box observations")
        if not isinstance(act_space, spaces.Discrete) or act_space.start != 0:
            raise ValueError("A3C requires a zero-based discrete action space")
        if device not in ("cpu", "auto"):
            raise ValueError("A3C shared-memory implementation supports only CPU")
        for name, value in (
            ("learning_rate", learning_rate),
            ("entropy_coef", entropy_coef),
            ("value_loss_coef", value_loss_coef),
            ("max_grad_norm", max_grad_norm),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name == "learning_rate" and value == 0)
            ):
                raise ValueError(f"invalid {name}")
        self.device = torch.device("cpu")
        self.obs_dim, self.act_dim = int(np.prod(obs_space.shape)), int(act_space.n)
        self.hidden_sizes, self.activation = tuple(hidden_sizes), activation
        self.entropy_coef, self.value_loss_coef = entropy_coef, value_loss_coef
        self.max_grad_norm, self.update_lock = max_grad_norm, update_lock
        self.last_learning_rate = torch.tensor(
            float(learning_rate), dtype=torch.float64
        ).share_memory_()
        self.model = ActorCritic(
            self.obs_dim, self.act_dim, self.hidden_sizes, activation
        )
        self.model.share_memory()
        self.optimizer = SharedAdam(self.model.parameters(), lr=learning_rate)
        self.optimizer.share_memory()

    def _lock(self):
        return self.update_lock if self.update_lock is not None else nullcontext()

    def new_local_model(self) -> ActorCritic:
        local = ActorCritic(
            self.obs_dim, self.act_dim, self.hidden_sizes, self.activation
        )
        self.sync_local(local)
        return local

    def sync_local(self, local_model: ActorCritic) -> None:
        with self._lock():
            local_model.load_state_dict(self.model.state_dict())

    def compute_loss(
        self,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropies: torch.Tensor,
        entropy_coef: Optional[float] = None,
        normalize_advantages: bool = False,
    ) -> Tuple[torch.Tensor, A3CStats]:
        tensors = (advantages, log_probs, values, returns, entropies)
        if not advantages.numel() or any(t.shape != advantages.shape for t in tensors):
            raise ValueError("loss inputs must have matching nonempty shapes")
        if any(not torch.isfinite(t).all() for t in tensors):
            raise ValueError("loss inputs must be finite")
        ent_coef = self.entropy_coef if entropy_coef is None else entropy_coef
        if not math.isfinite(ent_coef) or ent_coef < 0:
            raise ValueError("entropy coefficient must be finite and nonnegative")
        adv = advantages.detach()
        if normalize_advantages:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        policy_loss = -(adv * log_probs).mean()
        value_loss = F.mse_loss(values, returns.detach())
        entropy = entropies.mean()
        loss = policy_loss + self.value_loss_coef * value_loss - ent_coef * entropy
        if not torch.isfinite(loss):
            raise ValueError("A3C loss must be finite")
        stats = A3CStats(
            float(policy_loss.item()),
            float(value_loss.item()),
            float(entropy.item()),
            float(loss.item()),
            float(values.detach().mean().item()),
            float(returns.detach().mean().item()),
            float(advantages.detach().mean().item()),
        )
        return loss, stats

    def apply_gradients(
        self,
        local_model: ActorCritic,
        learning_rate: Optional[float] = None,
        sync: bool = True,
    ) -> bool:
        if learning_rate is not None and (
            not math.isfinite(learning_rate) or learning_rate < 0
        ):
            raise ValueError("learning rate must be finite and nonnegative")
        with self._lock():
            self.optimizer.zero_grad(set_to_none=True)
            gradients = [p.grad for p in local_model.parameters() if p.grad is not None]
            try:
                if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                    return False
                limit = self.max_grad_norm if self.max_grad_norm > 0 else float("inf")
                try:
                    clip_grad_norm_(
                        local_model.parameters(), limit, error_if_nonfinite=True
                    )
                except RuntimeError as error:
                    if (
                        "non-finite" not in str(error).lower()
                        and "nonfinite" not in str(error).lower()
                    ):
                        raise
                    return False
                for global_param, local_param in zip(
                    self.model.parameters(), local_model.parameters()
                ):
                    global_param.grad = (
                        None
                        if local_param.grad is None
                        else local_param.grad.detach().clone()
                    )
                if learning_rate is not None:
                    for group in self.optimizer.param_groups:
                        group["lr"] = learning_rate
                self.optimizer.step()
                self.last_learning_rate.fill_(
                    float(self.optimizer.param_groups[0]["lr"])
                )
                return True
            finally:
                self.optimizer.zero_grad(set_to_none=True)
                local_model.zero_grad(set_to_none=True)
                if sync:
                    local_model.load_state_dict(self.model.state_dict())
