from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from .networks import ActorCritic


@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes,
        activation: str,
        lr: float,
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        device: str,
    ):
        self.device = device
        self.model = ActorCritic(obs_dim, act_dim, hidden_sizes, activation).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def act(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action, logprob, entropy, value = self.model.act(obs_t)
        return action.cpu().numpy(), logprob.cpu().numpy(), value.cpu().numpy()

    def evaluate(self, obs_t: torch.Tensor, actions_t: torch.Tensor):
        logprob, entropy, value = self.model.evaluate_actions(obs_t, actions_t)
        return logprob, entropy, value

    def update(self, batch: Batch):
        obs, actions, old_logprobs, returns, advantages, old_values = (
            batch.obs,
            batch.actions,
            batch.logprobs,
            batch.returns,
            batch.advantages,
            batch.values,
        )
        logprob, entropy, value = self.evaluate(obs, actions)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        ratio = torch.exp(logprob - old_logprobs)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
        )
        policy_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

        # Value loss
        value_clipped = old_values + (value - old_values).clamp(
            -self.clip_coef, self.clip_coef
        )
        vf_losses1 = (value - returns) ** 2
        vf_losses2 = (value_clipped - returns) ** 2
        value_loss = 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))

        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        approx_kl = 0.5 * torch.mean((old_logprobs - logprob) ** 2).item()
        return {
            "loss/policy": policy_loss.item(),
            "loss/value": value_loss.item(),
            "loss/total": loss.item(),
            "stats/entropy": entropy.item(),
            "stats/approx_kl": approx_kl,
        }

    @staticmethod
    def compute_gae(
        rewards, dones, values, next_value, gamma: float, gae_lambda: float
    ):
        T, N = rewards.shape
        advantages = np.zeros((T, N), dtype=np.float32)
        lastgaelam = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns
