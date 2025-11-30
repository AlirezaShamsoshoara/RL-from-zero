from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from gymnasium import spaces
from IQL.iql.networks import GaussianPolicy, QNetwork, ValueNetwork


def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff >= 0, expectile, 1 - expectile)
    return weight * diff.pow(2)


@dataclass
class IQLStats:
    critic_loss: float
    value_loss: float
    actor_loss: float
    mean_advantage: float
    weight_mean: float
    weight_max: float


class IQLAgent:
    def __init__(
        self,
        obs_space: spaces.Box,
        act_space: spaces.Box,
        hidden_sizes,
        activation: str,
        actor_lr: float,
        critic_lr: float,
        value_lr: float,
        gamma: float,
        expectile: float,
        temperature: float,
        max_weight: float,
        tau: float,
        device: str,
    ):
        if not isinstance(obs_space, spaces.Box):
            raise TypeError("Observation space must be gym.spaces.Box")
        if not isinstance(act_space, spaces.Box):
            raise TypeError("Action space must be gym.spaces.Box")
        if len(obs_space.shape) != 1:
            raise ValueError("Only flat observation spaces are supported")
        if len(act_space.shape) != 1:
            raise ValueError("Only flat action spaces are supported")

        self.device = torch.device(device)
        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_dim = int(np.prod(act_space.shape))

        action_low = torch.as_tensor(act_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(act_space.high, dtype=torch.float32)

        self.actor = GaussianPolicy(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)

        self.q1 = QNetwork(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.q2 = QNetwork(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.value = ValueNetwork(
            obs_dim=self.obs_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.value_target = ValueNetwork(
            obs_dim=self.obs_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.value_target.load_state_dict(self.value.state_dict())

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.value_opt = Adam(self.value.parameters(), lr=value_lr)

        self.gamma = gamma
        self.expectile = expectile
        self.temperature = temperature
        self.max_weight = max_weight
        self.tau = tau
        self.global_step = 0

    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs[None, :]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(obs_t)
            else:
                action, _, _ = self.actor.sample(obs_t)
        return action.cpu().numpy()[0]

    def update(self, batch: Tuple[torch.Tensor, ...]) -> IQLStats:
        obs, actions, rewards, next_obs, dones = batch
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device).view(-1)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device).view(-1)

        with torch.no_grad():
            target_v = self.value_target(next_obs).squeeze(-1)
            target_q = rewards + (1.0 - dones) * self.gamma * target_v

        q1_pred = self.q1(obs, actions).squeeze(-1)
        q2_pred = self.q2(obs, actions).squeeze(-1)
        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        with torch.no_grad():
            min_q_detached = torch.min(
                self.q1(obs, actions), self.q2(obs, actions)
            ).squeeze(-1)
        value_pred = self.value(obs).squeeze(-1)
        value_loss = _expectile_loss(min_q_detached - value_pred, self.expectile).mean()

        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()

        self._soft_update_value()

        with torch.no_grad():
            v_detached = self.value(obs).squeeze(-1)
            advantages = torch.min(
                self.q1(obs, actions), self.q2(obs, actions)
            ).squeeze(-1) - v_detached
            temp = max(self.temperature, 1e-6)
            weights = torch.exp(advantages / temp)
            if self.max_weight > 0:
                weights = torch.clamp(weights, max=self.max_weight)

        log_prob = self.actor.log_prob(obs, actions).squeeze(-1)
        actor_loss = -(weights * log_prob).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.global_step += 1

        return IQLStats(
            critic_loss=float(critic_loss.item()),
            value_loss=float(value_loss.item()),
            actor_loss=float(actor_loss.item()),
            mean_advantage=float(advantages.mean().item()),
            weight_mean=float(weights.mean().item()),
            weight_max=float(weights.max().item()),
        )

    def _soft_update_value(self):
        for src, tgt in zip(self.value.parameters(), self.value_target.parameters()):
            tgt.data.copy_(self.tau * src.data + (1.0 - self.tau) * tgt.data)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "value": self.value.state_dict(),
            "value_target": self.value_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "value_opt": self.value_opt.state_dict(),
            "global_step": self.global_step,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        if "actor" in state:
            self.actor.load_state_dict(state["actor"])
        if "q1" in state:
            self.q1.load_state_dict(state["q1"])
        if "q2" in state:
            self.q2.load_state_dict(state["q2"])
        if "value" in state:
            self.value.load_state_dict(state["value"])
        if "value_target" in state:
            self.value_target.load_state_dict(state["value_target"])
        else:
            self.value_target.load_state_dict(self.value.state_dict())
        if "actor_opt" in state:
            self.actor_opt.load_state_dict(state["actor_opt"])
        if "critic_opt" in state:
            self.critic_opt.load_state_dict(state["critic_opt"])
        if "value_opt" in state:
            self.value_opt.load_state_dict(state["value_opt"])
        if "global_step" in state:
            self.global_step = int(state["global_step"])
