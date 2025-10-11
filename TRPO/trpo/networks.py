from __future__ import annotations
from typing import Iterable, Tuple
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
    act_key = (activation or "tanh").lower()
    if act_key not in act_cls:
        raise ValueError(f"Unsupported activation: {activation}")
    act_factory = act_cls[act_key]
    for hidden in hidden_sizes:
        layers.append(nn.Linear(last_dim, hidden))
        layers.append(act_factory())
        last_dim = hidden
    return nn.Sequential(*layers)


def _atanh(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


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
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        y = torch.tanh(z)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(z)
        log_prob = log_prob - torch.log(self.action_scale)
        log_prob = log_prob - torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, entropy, mean_action

    def log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, log_std = self(obs)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)

        y = (actions - self.action_bias) / self.action_scale
        y = torch.clamp(y, -0.999999, 0.999999)
        z = _atanh(y)
        log_prob = normal.log_prob(z)
        log_prob = log_prob - torch.log(self.action_scale)
        log_prob = log_prob - torch.log(1 - y.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)

    def kl_divergence(
        self,
        obs: torch.Tensor,
        old_mean: torch.Tensor,
        old_log_std: torch.Tensor,
    ) -> torch.Tensor:
        mean, log_std = self(obs)
        std = torch.exp(log_std)
        old_std = torch.exp(old_log_std)

        numerator = (old_mean - mean).pow(2) + old_std.pow(2) - std.pow(2)
        denominator = 2.0 * std.pow(2) + 1e-8
        kl = log_std - old_log_std + numerator / denominator
        return kl.sum(dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Iterable[int], activation: str):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, activation)
        last_hidden = hidden_sizes[-1] if hidden_sizes else obs_dim
        self.head = nn.Linear(last_hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.net(obs)
        return self.head(h).squeeze(-1)
