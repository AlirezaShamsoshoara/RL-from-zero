from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from gymnasium import spaces
from .networks import ActorCritic
from .utils import SharedAdam


@dataclass
class A3CStats:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float


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
    ) -> None:
        if not isinstance(obs_space, spaces.Box):
            raise ValueError("A3C requires a Box observation space")
        if len(obs_space.shape) != 1:
            raise ValueError("A3C supports only flat observation spaces")
        if not isinstance(act_space, spaces.Discrete):
            raise ValueError("A3C requires a discrete action space")
        if device != "cpu":
            raise ValueError("A3C shared-memory implementation currently supports only CPU device")

        self.device = torch.device("cpu")
        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_dim = int(act_space.n)
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        self.model = ActorCritic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        ).to(self.device)
        self.model.share_memory()

        self.optimizer = SharedAdam(self.model.parameters(), lr=learning_rate)
        self.optimizer.share_memory()

    def new_local_model(self) -> ActorCritic:
        local = ActorCritic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
        ).to(self.device)
        local.load_state_dict(self.model.state_dict())
        return local

    def sync_local(self, local_model: ActorCritic) -> None:
        local_model.load_state_dict(self.model.state_dict())

    def compute_loss(
        self,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropies: torch.Tensor,
    ) -> Tuple[torch.Tensor, A3CStats]:
        policy_loss = -(advantages.detach() * log_probs).mean()
        value_loss = F.mse_loss(values, returns)
        entropy = entropies.mean()
        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        stats = A3CStats(
            policy_loss=float(policy_loss.item()),
            value_loss=float(value_loss.item()),
            entropy=float(entropy.item()),
            total_loss=float(total_loss.item()),
        )
        return total_loss, stats

    def apply_gradients(self, local_model: ActorCritic) -> None:
        self.optimizer.zero_grad()
        for global_param, local_param in zip(self.model.parameters(), local_model.parameters()):
            if local_param.grad is None:
                continue
            global_param.grad = local_param.grad.detach().clone()
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        for param in local_model.parameters():
            param.grad = None
