from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from gymnasium import spaces
from SAC.sac.networks import GaussianPolicy, QNetwork


@dataclass
class SACStats:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha_value: float
    log_prob: float


class SACAgent:
    def __init__(
        self,
        obs_space: spaces.Box,
        act_space: spaces.Box,
        hidden_sizes,
        activation: str,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        target_entropy_scale: float,
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
        self.q1_target = QNetwork(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.q2_target = QNetwork(
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
        self.log_alpha = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32, device=self.device))
        self.alpha_opt = Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -float(self.act_dim) * target_entropy_scale

        self.gamma = gamma
        self.tau = tau
        self.global_step = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs[None, :]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(obs_t)
            else:
                action, _, _ = self.actor.sample(obs_t)
        return action.cpu().numpy()[0]

    def update(self, batch) -> SACStats:
        obs, actions, rewards, next_obs, dones = batch
        rewards = rewards.squeeze(-1)
        dones = dones.squeeze(-1)

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_obs)
            target_q1 = self.q1_target(next_obs, next_actions)
            target_q2 = self.q2_target(next_obs, next_actions)
            target_v = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = rewards + (1.0 - dones) * self.gamma * target_v

        current_q1 = self.q1(obs, actions)
        current_q2 = self.q2(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_actions, log_prob, _ = self.actor.sample(obs)
        q1_pi = self.q1(obs, new_actions)
        q2_pi = self.q2(obs, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        self.global_step += 1

        return SACStats(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha_value=float(self.alpha.item()),
            log_prob=float(log_prob.mean().item()),
        )

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module):
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.copy_(self.tau * src_p.data + (1.0 - self.tau) * tgt_p.data)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor_state_dict": self.actor.state_dict(),
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "q1_target_state_dict": self.q1_target.state_dict(),
            "q2_target_state_dict": self.q2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_opt.state_dict(),
            "critic_optimizer_state_dict": self.critic_opt.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "global_step": self.global_step,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        if "actor_state_dict" in state:
            self.actor.load_state_dict(state["actor_state_dict"])
            self.q1.load_state_dict(state["q1_state_dict"])
            self.q2.load_state_dict(state["q2_state_dict"])
            self.q1_target.load_state_dict(state.get("q1_target_state_dict", state["q1_state_dict"]))
            self.q2_target.load_state_dict(state.get("q2_target_state_dict", state["q2_state_dict"]))
        if "actor_optimizer_state_dict" in state:
            self.actor_opt.load_state_dict(state["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in state:
            self.critic_opt.load_state_dict(state["critic_optimizer_state_dict"])
        if "alpha_optimizer_state_dict" in state:
            self.alpha_opt.load_state_dict(state["alpha_optimizer_state_dict"])
        if "log_alpha" in state:
            log_alpha = state["log_alpha"]
            if not isinstance(log_alpha, torch.Tensor):
                log_alpha = torch.tensor(float(log_alpha), dtype=torch.float32)
            self.log_alpha.data.copy_(log_alpha.to(self.device))
        if "global_step" in state:
            self.global_step = int(state["global_step"])
