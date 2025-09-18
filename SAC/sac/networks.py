from __future__ import annotations
from typing import Iterable, Tuple
import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


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


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Iterable[int],
        activation: str,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        log_std_bounds: Tuple[float, float] = (-20.0, 2.0),
    ):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, activation)
        last_hidden = hidden_sizes[-1] if hidden_sizes else obs_dim
        self.mean_layer = nn.Linear(last_hidden, act_dim)
        self.log_std_layer = nn.Linear(last_hidden, act_dim)
        self.log_std_min, self.log_std_max = log_std_bounds

        action_scale = ((action_high - action_low) / 2.0).clamp(min=1e-6)
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mean = self.mean_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        y = torch.tanh(z)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(z)
        log_prob = log_prob - torch.log(self.action_scale)
        log_prob = log_prob - torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class QNetwork(nn.Module):
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
