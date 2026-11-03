from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn


def build_mlp(
    input_dim: int,
    hidden_sizes: Iterable[int],
    activation: str,
    layer_norm: bool = False,
) -> nn.Sequential:
    """Build a multi-layer perceptron.

    Optionally inserts a LayerNorm after each hidden linear layer (before the
    nonlinearity). LayerNorm is a well-established stabilizer for continuous
    control (used by D4PG, SAC) - it lets the critic fit smoother value
    surfaces without exploding activations, which is especially useful for the
    high-dimensional centralized critic input in MADDPG.
    """
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
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(act_factory())
        last_dim = hidden
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Actor network for MADDPG.
    Takes individual agent observation and outputs continuous action.
    Uses tanh activation to bound actions.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Iterable[int],
        activation: str,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        layer_norm: bool = False,
    ):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, activation, layer_norm=layer_norm)
        last_hidden = hidden_sizes[-1] if hidden_sizes else obs_dim
        self.head = nn.Linear(last_hidden, act_dim)
        # DDPG-standard small init on the final layer so the initial actor
        # output is near zero (before tanh). Without this, PyTorch's default
        # Kaiming init produces large pre-tanh values that saturate the tanh
        # (near +/-1), which gives near-zero gradient through it and cripples
        # early learning; it also biases initial exploration toward the
        # arbitrary saturation direction of the random init.
        nn.init.uniform_(self.head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.head.bias, -3e-3, 3e-3)

        # Action scaling: maps tanh output [-1, 1] to [action_low, action_high]
        action_scale = ((action_high - action_low) / 2.0).clamp(min=1e-6)
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through actor network.
        Args:
            obs: Agent observation [batch_size, obs_dim]
        Returns:
            action: Continuous action [batch_size, act_dim]
        """
        h = self.net(obs)
        action = torch.tanh(self.head(h))
        return action * self.action_scale + self.action_bias


class Critic(nn.Module):
    """
    Centralized critic for MADDPG.
    Takes all agents' observations and actions as input.
    Outputs Q-value for the given agent.
    """

    def __init__(
        self,
        total_obs_dim: int,
        total_act_dim: int,
        hidden_sizes: Iterable[int],
        activation: str,
        layer_norm: bool = False,
    ):
        super().__init__()
        # Critic input is concatenation of all observations and all actions
        self.net = build_mlp(total_obs_dim + total_act_dim, hidden_sizes, activation,
                             layer_norm=layer_norm)
        last_hidden = hidden_sizes[-1] if hidden_sizes else total_obs_dim + total_act_dim
        self.head = nn.Linear(last_hidden, 1)
        # DDPG-standard small init on the final layer: initial Q-value is
        # near zero so the actor gets an unbiased gradient signal early on.
        nn.init.uniform_(self.head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.head.bias, -3e-3, 3e-3)

    def forward(self, all_obs: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network.
        Args:
            all_obs: Concatenated observations from all agents [batch_size, total_obs_dim]
            all_actions: Concatenated actions from all agents [batch_size, total_act_dim]
        Returns:
            q_value: Q-value [batch_size, 1]
        """
        x = torch.cat([all_obs, all_actions], dim=-1)
        h = self.net(x)
        return self.head(h)
