from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from gym import spaces
from TD3.td3.networks import Actor, Critic


@dataclass
class TD3Stats:
    critic_loss: float
    actor_loss: Optional[float]
    q1_value: float
    q2_value: float


class TD3Agent:
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
        policy_delay: int,
        target_noise: float,
        noise_clip: float,
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

        self.q1 = Critic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.q2 = Critic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.q1_target = Critic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.q2_target = Critic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )

        self.gamma = gamma
        self.tau = tau
        self.policy_delay = max(1, int(policy_delay))
        self.target_noise = float(target_noise)
        self.noise_clip = float(noise_clip)
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

    def update(self, batch) -> TD3Stats:
        obs, actions, rewards, next_obs, dones = batch
        rewards = rewards
        dones = dones

        with torch.no_grad():
            noise = (
                torch.randn_like(actions) * self.target_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_obs) + noise).clamp(
                self.action_low, self.action_high
            )
            target_q1 = self.q1_target(next_obs, next_actions)
            target_q2 = self.q2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1 = self.q1(obs, actions)
        current_q2 = self.q2(obs, actions)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss_value: Optional[float] = None
        self.global_step += 1
        if self.global_step % self.policy_delay == 0:
            pi_actions = self.actor(obs)
            actor_loss = -self.q1(obs, pi_actions).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss_value = float(actor_loss.item())

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
        else:
            actor_loss_value = None

        return TD3Stats(
            critic_loss=float(critic_loss.item()),
            actor_loss=actor_loss_value,
            q1_value=float(current_q1.mean().item()),
            q2_value=float(current_q2.mean().item()),
        )

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module):
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.copy_(self.tau * src_p.data + (1.0 - self.tau) * tgt_p.data)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "q1_target_state_dict": self.q1_target.state_dict(),
            "q2_target_state_dict": self.q2_target.state_dict(),
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
        if "q1_state_dict" in state:
            self.q1.load_state_dict(state["q1_state_dict"])
        if "q2_state_dict" in state:
            self.q2.load_state_dict(state["q2_state_dict"])
        if "q1_target_state_dict" in state:
            self.q1_target.load_state_dict(state["q1_target_state_dict"])
        else:
            self.q1_target.load_state_dict(self.q1.state_dict())
        if "q2_target_state_dict" in state:
            self.q2_target.load_state_dict(state["q2_target_state_dict"])
        else:
            self.q2_target.load_state_dict(self.q2.state_dict())
        if "actor_optimizer_state_dict" in state:
            self.actor_opt.load_state_dict(state["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in state:
            self.critic_opt.load_state_dict(state["critic_optimizer_state_dict"])
        if "global_step" in state:
            self.global_step = int(state["global_step"])
