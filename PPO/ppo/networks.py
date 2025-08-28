from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str):
    name = (name or "tanh").lower()
    if name == "relu":
        return nn.ReLU()
    return nn.Tanh()


class ActorCritic(nn.Module):
    def __init__(
        self, obs_dim: int, act_dim: int, hidden_sizes=(64, 64), activation="tanh"
    ):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), get_activation(activation)]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.pi = nn.Linear(last, act_dim)
        self.v = nn.Linear(last, 1)

    def forward(self, x):
        x = self.backbone(x)
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action, logprob, entropy, value

    def evaluate_actions(self, obs, actions):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return logprob, entropy, value
