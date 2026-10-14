from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple

from .normalization import RunningNorm


def get_activation(name: str):
    name = (name or "tanh").lower()
    if name == "relu":
        return nn.ReLU()
    return nn.Tanh()


class Actor(nn.Module):
    """
    Discrete actor network for MAPPO.
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

    def get_action(self, obs, deterministic: bool = False):
        """Sample action from policy distribution."""
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
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


class GaussianActor(nn.Module):
    """
    Continuous actor network with tanh-squashed Gaussian policy.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes=(256, 128),
        activation="relu",
        action_low=None,
        action_high=None,
        log_std_init: float = -0.5,
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
    ):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), get_activation(activation)]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(last, act_dim)

        self.log_std = nn.Parameter(torch.full((act_dim,), log_std_init))
        self.log_std_min, self.log_std_max = log_std_bounds

        if action_low is None or action_high is None:
            low = torch.full((act_dim,), -1.0)
            high = torch.full((act_dim,), 1.0)
        else:
            low = torch.as_tensor(action_low, dtype=torch.float32)
            high = torch.as_tensor(action_high, dtype=torch.float32)

        if (
            low.shape != (act_dim,)
            or high.shape != (act_dim,)
            or not torch.isfinite(low).all()
            or not torch.isfinite(high).all()
            or not torch.all(high > low)
        ):
            raise ValueError("continuous action bounds must be finite, ordered vectors")
        self.register_buffer("action_scale", high * 0.5 - low * 0.5)
        self.register_buffer("action_bias", high * 0.5 + low * 0.5)

    def forward(self, obs):
        x = self.backbone(obs)
        mean = self.mu(x)
        return mean

    def _distribution(self, obs):
        mean = self.forward(obs)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        log_std = log_std.expand_as(mean)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)

    def _logprob(self, dist, unit_action):
        unit_action = unit_action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(unit_action)
        jacobian = torch.log1p(-unit_action.square()) + self.action_scale.log()
        return (dist.log_prob(raw) - jacobian).sum(dim=-1)

    def get_action(self, obs, deterministic: bool = False):
        dist = self._distribution(obs)
        raw_action = dist.mean if deterministic else dist.sample()
        unit_action = torch.tanh(raw_action).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        action = unit_action * self.action_scale + self.action_bias
        logprob, entropy = self.evaluate_actions(obs, action)
        return action, logprob, entropy

    def evaluate_actions(self, obs, actions):
        dist = self._distribution(obs)
        unit_action = (actions - self.action_bias) / self.action_scale
        if not torch.isfinite(actions).all() or torch.any(
            unit_action.abs() > 1.0 + 1e-5
        ):
            raise ValueError(
                "continuous actions must be finite and within policy bounds"
            )
        # Entropy regularization uses the unsquashed Gaussian entropy proxy.
        return self._logprob(dist, unit_action), dist.entropy().sum(dim=-1)


class CentralizedCritic(nn.Module):
    """
    Centralized value function for MAPPO.
    Takes concatenated observations from all agents (full state) and outputs value.
    """

    def __init__(self, state_dim: int, hidden_sizes=(256, 128), activation="relu"):
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

    def __init__(self, obs_dim: int, hidden_sizes=(256, 128), activation="relu"):
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
        action_type: str = "discrete",
        action_low=None,
        action_high=None,
        normalize_inputs: bool = False,
    ):
        super().__init__()
        self.use_centralized_critic = use_centralized_critic
        if action_type not in {"continuous", "discrete"}:
            raise ValueError("action_type must be continuous or discrete")
        self.action_type = action_type
        if action_type == "continuous":
            self.actor = GaussianActor(
                obs_dim,
                act_dim,
                actor_hidden_sizes,
                activation,
                action_low=action_low,
                action_high=action_high,
            )
        else:
            self.actor = Actor(obs_dim, act_dim, actor_hidden_sizes, activation)
        self.normalize_inputs = normalize_inputs
        self.obs_norm = RunningNorm(obs_dim) if normalize_inputs else nn.Identity()
        self.state_norm = (
            RunningNorm(state_dim if use_centralized_critic else obs_dim)
            if normalize_inputs
            else nn.Identity()
        )
        if use_centralized_critic:
            self.critic = CentralizedCritic(state_dim, critic_hidden_sizes, activation)
        else:
            self.critic = DecentralizedCritic(obs_dim, critic_hidden_sizes, activation)

    def act(self, obs, deterministic: bool = False):
        """
        Sample action from policy.
        Args:
            obs: Individual agent observation
        Returns:
            action, logprob, entropy
        """
        return self.actor.get_action(self.obs_norm(obs), deterministic=deterministic)

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions for policy gradient.
        Args:
            obs: Individual agent observation
            actions: Actions taken
        Returns:
            logprob, entropy
        """
        return self.actor.evaluate_actions(self.obs_norm(obs), actions)

    def get_value(self, state_or_obs):
        """
        Compute value function.
        Args:
            state_or_obs: Full state (if centralized) or individual obs (if decentralized)
        Returns:
            value
        """
        return self.critic(self.state_norm(state_or_obs))
