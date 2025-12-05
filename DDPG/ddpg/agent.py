from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from gymnasium import spaces
from DDPG.ddpg.networks import Actor, Critic


@dataclass
class DDPGStats:
    critic_loss: float
    actor_loss: float
    q_value: float


class DDPGAgent:
    def __init__(
        self,
        obs_space: spaces.Box,
        act_space: spaces.Box,
        hidden_sizes,
        activation: str,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        target_policy_noise: float,
        target_noise_clip: float,
        device: str,
    ):
        assert isinstance(obs_space, spaces.Box), "Observation space must be Box"
        assert isinstance(act_space, spaces.Box), "Action space must be Box"
        assert len(obs_space.shape) == 1, "Only flat observation spaces supported"
        assert len(act_space.shape) == 1, "Only flat action spaces supported"

        self.device = torch.device(device)
        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_dim = int(np.prod(act_space.shape))

        action_low = torch.as_tensor(act_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(act_space.high, dtype=torch.float32)
        self.action_low = action_low.to(self.device)
        self.action_high = action_high.to(self.device)
        self.action_low_np = action_low.cpu().numpy()
        self.action_high_np = action_high.cpu().numpy()

        self.actor = Actor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)
        self.actor_target = Actor(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.critic_target = Critic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.global_step = 0

    def act(self, obs: np.ndarray, noise: float = 0.0, deterministic: bool = False) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs[None, :]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        if deterministic or noise <= 0.0:
            return action
        noise_sample = np.random.normal(0.0, noise, size=action.shape)
        action = action + noise_sample
        return np.clip(action, self.action_low_np, self.action_high_np)

    def update(self, batch) -> DDPGStats:
        obs, actions, rewards, next_obs, dones = batch

        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            if self.target_policy_noise > 0.0:
                noise = torch.randn_like(next_actions) * self.target_policy_noise
                if self.target_noise_clip > 0.0:
                    noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = next_actions + noise
            next_actions = next_actions.clamp(self.action_low, self.action_high)
            target_q = self.critic_target(next_obs, next_actions)
            target = rewards + (1.0 - dones) * self.gamma * target_q

        current_q = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        pi_actions = self.actor(obs)
        actor_loss = -self.critic(obs, pi_actions).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.global_step += 1

        return DDPGStats(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            q_value=float(current_q.mean().item()),
        )

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module):
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.copy_(self.tau * src_p.data + (1.0 - self.tau) * tgt_p.data)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_opt.state_dict(),
            "critic_optimizer_state_dict": self.critic_opt.state_dict(),
            "global_step": self.global_step,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        if "actor_state_dict" in state:
            self.actor.load_state_dict(state["actor_state_dict"])
        if "actor_target_state_dict" in state:
            self.actor_target.load_state_dict(state["actor_target_state_dict"])
        else:
            self.actor_target.load_state_dict(self.actor.state_dict())
        if "critic_state_dict" in state:
            self.critic.load_state_dict(state["critic_state_dict"])
        if "critic_target_state_dict" in state:
            self.critic_target.load_state_dict(state["critic_target_state_dict"])
        else:
            self.critic_target.load_state_dict(self.critic.state_dict())
        if "actor_optimizer_state_dict" in state:
            self.actor_opt.load_state_dict(state["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in state:
            self.critic_opt.load_state_dict(state["critic_optimizer_state_dict"])
        if "global_step" in state:
            self.global_step = int(state["global_step"])

