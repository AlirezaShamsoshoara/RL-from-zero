from __future__ import annotations

from numbers import Integral
from typing import Iterable

import torch
import torch.nn as nn


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _hidden_sizes(values: Iterable[int]) -> tuple[int, ...]:
    try:
        sizes = tuple(values)
    except TypeError as exc:
        raise ValueError(
            "hidden_sizes must be an iterable of positive integers"
        ) from exc
    return tuple(_positive_int(size, "hidden size") for size in sizes)


def _action_bounds(low, high, act_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy finite, strictly ordered bounds representable in float32."""
    bounds = []
    for value, name in ((low, "action_low"), (high, "action_high")):
        raw = torch.as_tensor(value)
        if raw.is_complex() or raw.dtype == torch.bool:
            raise ValueError(f"{name} must contain real numeric bounds")
        if raw.shape != (act_dim,) or not torch.isfinite(raw).all():
            raise ValueError(f"{name} must be finite with shape ({act_dim},)")
        bound = raw.detach().to(dtype=torch.float32).clone()
        if not torch.isfinite(bound).all():
            raise ValueError(f"{name} must remain finite in float32")
        bounds.append(bound)
    low, high = bounds
    if low.device != high.device or not (low < high).all():
        raise ValueError("action bounds must share a device and satisfy low < high")
    # Half first avoids overflow in high - low and high + low.
    scale = high / 2.0 - low / 2.0
    bias = high / 2.0 + low / 2.0
    if (
        not torch.isfinite(scale).all()
        or not torch.isfinite(bias).all()
        or not (scale > 0).all()
    ):
        raise ValueError("action bounds must have a finite positive float32 scale")
    return low, high


def _input_tensor(value: torch.Tensor, width: int, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim not in (1, 2) or value.shape[-1] != width:
        raise ValueError(f"{name} must have shape ({width},) or (N, {width})")
    if value.ndim == 2 and value.shape[0] == 0:
        raise ValueError(f"{name} batch must not be empty")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def build_mlp(
    input_dim: int, hidden_sizes: Iterable[int], activation: str
) -> nn.Sequential:
    input_dim = _positive_int(input_dim, "input_dim")
    sizes = _hidden_sizes(hidden_sizes)
    factories = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": lambda: nn.LeakyReLU(0.01),
        "elu": nn.ELU,
    }
    if not isinstance(activation, str) or activation.lower() not in factories:
        raise ValueError(f"Unsupported activation: {activation}")
    layers = []
    last_dim = input_dim
    for hidden in sizes:
        layers.extend((nn.Linear(last_dim, hidden), factories[activation.lower()]()))
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
        self.obs_dim = _positive_int(obs_dim, "obs_dim")
        self.act_dim = _positive_int(act_dim, "act_dim")
        sizes = _hidden_sizes(hidden_sizes)
        low, high = _action_bounds(action_low, action_high, self.act_dim)
        self.net = build_mlp(self.obs_dim, sizes, activation)
        self.head = nn.Linear(sizes[-1] if sizes else self.obs_dim, self.act_dim)
        self.register_buffer("action_scale", high / 2.0 - low / 2.0)
        self.register_buffer("action_bias", high / 2.0 + low / 2.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        _input_tensor(obs, self.obs_dim, "obs")
        if obs.device != self.head.weight.device or obs.dtype != self.head.weight.dtype:
            raise ValueError("obs must match the actor's device and dtype")
        action = torch.tanh(self.head(self.net(obs)))
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
        self.obs_dim = _positive_int(obs_dim, "obs_dim")
        self.act_dim = _positive_int(act_dim, "act_dim")
        sizes = _hidden_sizes(hidden_sizes)
        self.net = build_mlp(self.obs_dim + self.act_dim, sizes, activation)
        self.head = nn.Linear(sizes[-1] if sizes else self.obs_dim + self.act_dim, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        _input_tensor(obs, self.obs_dim, "obs")
        _input_tensor(act, self.act_dim, "act")
        if obs.shape[:-1] != act.shape[:-1]:
            raise ValueError("obs and act must have matching batch shapes")
        if (
            obs.device != self.head.weight.device
            or act.device != obs.device
            or obs.dtype != self.head.weight.dtype
            or act.dtype != obs.dtype
        ):
            raise ValueError("obs and act must match the critic's device and dtype")
        return self.head(self.net(torch.cat((obs, act), dim=-1)))
