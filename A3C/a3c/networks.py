from __future__ import annotations
from typing import Iterable
import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    name = (name or "tanh").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    return nn.Tanh()


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Iterable[int] = (128, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers = []
        last_dim = obs_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(get_activation(activation))
            last_dim = hidden
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, act_dim)
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, obs: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.backbone(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value
