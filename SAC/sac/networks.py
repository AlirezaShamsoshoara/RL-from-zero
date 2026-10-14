from __future__ import annotations

import math
from numbers import Integral
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _hidden_sizes(hidden_sizes: Iterable[int]) -> tuple[int, ...]:
    try:
        sizes = tuple(hidden_sizes)
    except TypeError as exc:
        raise ValueError(
            "hidden_sizes must be an iterable of positive integers"
        ) from exc
    return tuple(_positive_integer(size, "hidden size") for size in sizes)


def get_activation(name: str) -> nn.Module:
    if not isinstance(name, str):
        raise ValueError("activation must be a supported string")
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


def build_mlp(
    input_dim: int, hidden_sizes: Iterable[int], activation: str
) -> nn.Sequential:
    last_dim = _positive_integer(input_dim, "input_dim")
    sizes = _hidden_sizes(hidden_sizes)
    get_activation(activation)  # Validate even an empty (linear-only) architecture.
    layers = []
    for hidden in sizes:
        layers.append(nn.Linear(last_dim, hidden))
        layers.append(get_activation(activation))
        last_dim = hidden
    return nn.Sequential(*layers)


def _validate_input(value: torch.Tensor, dim: int, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim not in (1, 2) or value.shape[-1] != dim or value.numel() == 0:
        raise ValueError(f"{name} must have shape ({dim},) or (N, {dim}) with N > 0")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _finite_output(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {name}")


class GaussianPolicy(nn.Module):
    """Tanh Gaussian policy whose log density is in physical action units."""

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
        self.obs_dim = _positive_integer(obs_dim, "obs_dim")
        self.act_dim = _positive_integer(act_dim, "act_dim")
        sizes = _hidden_sizes(hidden_sizes)
        get_activation(activation)
        try:
            low = torch.as_tensor(action_low)
            high = torch.as_tensor(action_high)
            if (
                low.is_complex()
                or high.is_complex()
                or low.dtype == torch.bool
                or high.dtype == torch.bool
            ):
                raise ValueError("Action bounds must contain real numeric values")
            low = low.detach().to(dtype=torch.float32).clone()
            high = high.detach().to(dtype=torch.float32).clone()
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError("Action bounds must be finite real vectors") from exc
        if low.shape != (self.act_dim,) or high.shape != (self.act_dim,):
            raise ValueError("Action bounds must have shape (act_dim,)")
        if low.device != high.device:
            raise ValueError("Action bounds must be on the same device")
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError("Action bounds must be finite in float32")
        width = high - low
        action_scale = width / 2.0
        action_bias = high / 2.0 + low / 2.0
        if (
            not torch.isfinite(width).all()
            or not torch.isfinite(action_scale).all()
            or not (action_scale > 0).all()
            or not torch.isfinite(action_bias).all()
        ):
            raise ValueError("Action bounds must be ordered with finite positive width")
        try:
            bounds = tuple(log_std_bounds)
            if len(bounds) != 2 or any(isinstance(x, bool) for x in bounds):
                raise ValueError(
                    "log_std_bounds must contain two ordered finite values"
                )
            lower, upper = (float(x) for x in bounds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "log_std_bounds must contain two ordered finite values"
            ) from exc
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("log_std_bounds must contain two ordered finite values")
        std_limits = torch.tensor([lower, upper], dtype=torch.float32).exp()
        if not torch.isfinite(std_limits).all() or not (std_limits > 0).all():
            raise ValueError("log_std_bounds must give finite positive float32 std")

        self.net = build_mlp(self.obs_dim, sizes, activation)
        last_hidden = sizes[-1] if sizes else self.obs_dim
        self.mean_layer = nn.Linear(last_hidden, self.act_dim)
        self.log_std_layer = nn.Linear(last_hidden, self.act_dim)
        self.log_std_min, self.log_std_max = lower, upper
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, obs: torch.Tensor):
        _validate_input(obs, self.obs_dim, "obs")
        h = self.net(obs)
        mean = self.mean_layer(h)
        raw_log_std = self.log_std_layer(h)
        _finite_output(mean, "policy mean")
        _finite_output(raw_log_std, "policy log std")
        log_std = torch.clamp(raw_log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs: torch.Tensor):
        mean, log_std = self(obs)
        std = log_std.exp()
        _finite_output(std, "policy std")
        if not (std > 0).all():
            raise FloatingPointError("Policy std must be positive")
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        _finite_output(z, "Gaussian sample")
        action = torch.tanh(z) * self.action_scale + self.action_bias
        # Exact log(1 - tanh(z)^2), stable even when tanh rounds to +/-1.
        log_tanh_jacobian = 2.0 * (math.log(2.0) - z - F.softplus(-2.0 * z))
        # Standardizing avoids the variance underflow in Normal.log_prob for
        # valid very small standard deviations. Keep the affine Jacobian:
        # these are densities of physical actions, not normalized tanh values.
        gaussian_log_prob = -0.5 * ((z - mean) / std).square() - log_std
        gaussian_log_prob = gaussian_log_prob - 0.5 * math.log(2.0 * math.pi)
        log_prob = (
            gaussian_log_prob - torch.log(self.action_scale) - log_tanh_jacobian
        ).sum(dim=-1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        _finite_output(action, "sampled action")
        _finite_output(log_prob, "policy log probability")
        _finite_output(mean_action, "mean action")
        return action, log_prob, mean_action

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        # No Normal construction or sampling on this path.
        mean, _ = self(obs)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        _finite_output(action, "deterministic action")
        return action


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Iterable[int],
        activation: str,
    ):
        super().__init__()
        self.obs_dim = _positive_integer(obs_dim, "obs_dim")
        self.act_dim = _positive_integer(act_dim, "act_dim")
        sizes = _hidden_sizes(hidden_sizes)
        self.net = build_mlp(self.obs_dim + self.act_dim, sizes, activation)
        last_hidden = sizes[-1] if sizes else self.obs_dim + self.act_dim
        self.head = nn.Linear(last_hidden, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        _validate_input(obs, self.obs_dim, "obs")
        _validate_input(act, self.act_dim, "act")
        if obs.shape[:-1] != act.shape[:-1]:
            raise ValueError("obs and act must have matching batch shapes")
        if obs.device != act.device or obs.dtype != act.dtype:
            raise ValueError("obs and act must have matching device and dtype")
        x = torch.cat([obs, act], dim=-1)
        value = self.head(self.net(x))
        _finite_output(value, "Q value")
        return value
