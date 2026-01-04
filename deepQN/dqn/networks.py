from __future__ import annotations
from typing import Iterable
import math
import torch
from torch import nn


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "gelu":
        return nn.GELU()
    return nn.ReLU()


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int],
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = obs_dim
        act = _get_activation(activation)
        for size in hidden_sizes:
            linear = nn.Linear(input_dim, size)
            self._init_layer(linear)
            layers.append(linear)
            layers.append(act.__class__())
            input_dim = size
        out_layer = nn.Linear(input_dim, action_dim)
        nn.init.orthogonal_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)
        self.model = nn.Sequential(*layers)

    @staticmethod
    def _init_layer(layer: nn.Linear) -> None:
        nn.init.orthogonal_(layer.weight)
        if layer.bias is not None:
            fan_in = layer.weight.size(1)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(layer.bias, -bound, bound)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)
