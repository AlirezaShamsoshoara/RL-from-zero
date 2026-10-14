from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    choices = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }
    if name not in choices:
        raise ValueError(f"unsupported activation: {name}")
    return choices[name]()


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Iterable[int] = (128, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        for name, value in (("obs_dim", obs_dim), ("act_dim", act_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.obs_dim, self.act_dim = obs_dim, act_dim
        layers = []
        last_dim = obs_dim
        for hidden in hidden_sizes:
            if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden <= 0:
                raise ValueError("hidden sizes must be positive integers")
            layers.extend((nn.Linear(last_dim, hidden), get_activation(activation)))
            last_dim = hidden
        get_activation(activation)
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, act_dim)
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, obs: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if (
            obs.dim() != 2
            or obs.shape[-1] != self.obs_dim
            or not torch.isfinite(obs).all()
        ):
            raise ValueError(
                "observations must be a finite vector or batch with obs_dim features"
            )
        features = self.backbone(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        if actions.shape != logits.shape[:-1] or actions.dtype not in (
            torch.int64,
            torch.int32,
        ):
            raise ValueError("actions must be an integer batch matching observations")
        if torch.any(actions < 0) or torch.any(actions >= self.act_dim):
            raise ValueError("actions outside the discrete action space")
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
