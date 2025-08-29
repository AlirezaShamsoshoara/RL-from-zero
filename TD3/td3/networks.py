from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn


def build_mlp(input_dim: int, hidden_sizes: Iterable[int], activation: str) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    act_cls = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": lambda: nn.LeakyReLU(0.01),
        "elu": nn.ELU,
    }
    if activation.lower() not in act_cls:
        raise ValueError(f"Unsupported activation: {activation}")
    act_factory = act_cls[activation.lower()]
    for hidden in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden))
        layers.append(act_factory())
        last_dim = hidden
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Iterable[int],
        activation: str,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
    ):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, activation)
        last_hidden = hidden_sizes[-1] if hidden_sizes else obs_dim
        self.head = nn.Linear(last_hidden, act_dim)
        action_scale = ((action_high - action_low) / 2.0).clamp(min=1e-6)
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.net(obs)
        action = torch.tanh(self.head(h))
        return action * self.action_scale + self.action_bias


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Iterable[int],
        activation: str,
    ):
        super().__init__()
        self.net = build_mlp(obs_dim + act_dim, hidden_sizes, activation)
        last_hidden = hidden_sizes[-1] if hidden_sizes else obs_dim + act_dim
        self.head = nn.Linear(last_hidden, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        h = self.net(x)
        return self.head(h)
