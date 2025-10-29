from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def get_activation(name: str):
    name = (name or "tanh").lower()
    if name == "relu":
        return nn.ReLU()
    return nn.Tanh()


class Actor(nn.Module):
    """
    Actor network for MAPPO.
    Takes individual agent observation and outputs action distribution.
    """

    def __init__(
        self, obs_dim: int, act_dim: int, hidden_sizes=(256, 128), activation="relu"
    ):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), get_activation(activation)]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.pi = nn.Linear(last, act_dim)

    def forward(self, obs):
        x = self.backbone(obs)
        logits = self.pi(x)
        return logits

    def get_action(self, obs):
        """Sample action from policy distribution."""
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy

    def evaluate_actions(self, obs, actions):
        """Evaluate log probabilities and entropy for given actions."""
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logprob = dist.log_prob(actions)
        entropy = dist.entropy()
        return logprob, entropy


class CentralizedCritic(nn.Module):
    """
    Centralized value function for MAPPO.
    Takes concatenated observations from all agents (full state) and outputs value.
    """

    def __init__(
        self, state_dim: int, hidden_sizes=(256, 128), activation="relu"
    ):
        super().__init__()
        layers = []
        last = state_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), get_activation(activation)]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.v = nn.Linear(last, 1)

    def forward(self, state):
        """
        Args:
            state: Full state (concatenated observations from all agents)
        Returns:
            value: State value
        """
        x = self.backbone(state)
        value = self.v(x).squeeze(-1)
        return value


class DecentralizedCritic(nn.Module):
    """
    Decentralized value function for MAPPO.
    Takes individual agent observation and outputs value.
    """

    def __init__(
        self, obs_dim: int, hidden_sizes=(256, 128), activation="relu"
    ):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), get_activation(activation)]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.v = nn.Linear(last, 1)

    def forward(self, obs):
        """
        Args:
            obs: Individual agent observation
        Returns:
            value: State value from agent's perspective
        """
        x = self.backbone(obs)
        value = self.v(x).squeeze(-1)
        return value


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for a single agent in MAPPO.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        state_dim: int,
        actor_hidden_sizes=(256, 128),
        critic_hidden_sizes=(256, 128),
        activation="relu",
        use_centralized_critic=True,
    ):
        super().__init__()
        self.use_centralized_critic = use_centralized_critic
        self.actor = Actor(obs_dim, act_dim, actor_hidden_sizes, activation)
        if use_centralized_critic:
            self.critic = CentralizedCritic(state_dim, critic_hidden_sizes, activation)
        else:
            self.critic = DecentralizedCritic(obs_dim, critic_hidden_sizes, activation)

    def act(self, obs):
        """
        Sample action from policy.
        Args:
            obs: Individual agent observation
        Returns:
            action, logprob, entropy
        """
        return self.actor.get_action(obs)

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions for policy gradient.
        Args:
            obs: Individual agent observation
            actions: Actions taken
        Returns:
            logprob, entropy
        """
        return self.actor.evaluate_actions(obs, actions)

    def get_value(self, state_or_obs):
        """
        Compute value function.
        Args:
            state_or_obs: Full state (if centralized) or individual obs (if decentralized)
        Returns:
            value
        """
        return self.critic(state_or_obs)
