from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .networks import ActorCritic


@dataclass
class Batch:
    obs: torch.Tensor  # [batch_size, obs_dim]
    state: torch.Tensor  # [batch_size, state_dim] for centralized critic
    actions: torch.Tensor  # [batch_size] or [batch_size, act_dim]
    logprobs: torch.Tensor  # [batch_size]
    returns: torch.Tensor  # [batch_size]
    advantages: torch.Tensor  # [batch_size]
    values: torch.Tensor  # [batch_size]
    alive_mask: torch.Tensor  # [batch_size]


class MAPPOAgent:
    """
    Multi-Agent PPO (MAPPO) implementation.
    Each agent has its own actor-critic network (or shared if share_policy=True).
    Supports centralized critic using full state information.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        state_dim: int,
        actor_hidden_sizes,
        critic_hidden_sizes,
        activation: str,
        lr: float,
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        device: str,
        share_policy: bool = False,
        use_centralized_critic: bool = True,
        action_type: str = "discrete",
        action_low=None,
        action_high=None,
    ):
        self.n_agents = n_agents
        self.device = device
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.share_policy = share_policy
        self.use_centralized_critic = use_centralized_critic

        # Create actor-critic networks for each agent
        if share_policy:
            # All agents share the same policy network
            shared_model = ActorCritic(
                obs_dim,
                act_dim,
                state_dim,
                actor_hidden_sizes,
                critic_hidden_sizes,
                activation,
                use_centralized_critic,
                action_type=action_type,
                action_low=action_low,
                action_high=action_high,
            ).to(device)
            self.models = [shared_model for _ in range(n_agents)]
            # Single optimizer for shared parameters
            self.optimizers = [torch.optim.Adam(shared_model.parameters(), lr=lr)]
        else:
            # Each agent has its own policy network
            self.models = [
                ActorCritic(
                    obs_dim,
                    act_dim,
                    state_dim,
                    actor_hidden_sizes,
                    critic_hidden_sizes,
                    activation,
                    use_centralized_critic,
                    action_type=action_type,
                    action_low=action_low,
                    action_high=action_high,
                ).to(device)
                for _ in range(n_agents)
            ]
            # Separate optimizer for each agent
            self.optimizers = [
                torch.optim.Adam(model.parameters(), lr=lr) for model in self.models
            ]

    @torch.no_grad()
    def act(
        self, obs_list: List[np.ndarray], state: np.ndarray = None, deterministic: bool = False
    ):
        """
        Sample actions for all agents.
        Args:
            obs_list: List of observations, one per agent [n_agents, obs_dim]
            state: Full state for centralized critic [state_dim] (optional)
        Returns:
            actions: [n_agents]
            logprobs: [n_agents]
            values: [n_agents]
        """
        actions = []
        logprobs = []
        values = []

        for i, obs in enumerate(obs_list):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, logprob, _ = self.models[i].act(obs_t, deterministic=deterministic)

            # Get value based on centralized or decentralized critic
            if self.use_centralized_critic and state is not None:
                state_t = torch.as_tensor(
                    state, dtype=torch.float32, device=self.device
                )
                value = self.models[i].get_value(state_t)
            else:
                value = self.models[i].get_value(obs_t)

            actions.append(action.cpu().numpy())
            logprobs.append(logprob.cpu().numpy())
            values.append(value.cpu().numpy())

        return np.array(actions), np.array(logprobs), np.array(values)

    def update(self, batches: List[Batch]) -> Tuple[Dict[str, float], int]:
        """
        Update policy and value networks for all agents.
        Args:
            batches: List of batches, one per agent
        Returns:
            Dictionary of loss statistics and number of agent updates performed
        """
        total_stats = {
            "loss/policy": 0.0,
            "loss/value": 0.0,
            "loss/total": 0.0,
            "stats/entropy": 0.0,
            "stats/approx_kl": 0.0,
        }
        update_steps = 0

        # Update each agent's policy
        for agent_id, batch in enumerate(batches):
            obs, state, actions, old_logprobs, returns, advantages, old_values, alive_mask = (
                batch.obs,
                batch.state,
                batch.actions,
                batch.logprobs,
                batch.returns,
                batch.advantages,
                batch.values,
                batch.alive_mask,
            )

            alive_idx = alive_mask > 0.5
            if not torch.any(alive_idx):
                continue

            obs = obs[alive_idx]
            state = state[alive_idx]
            actions = actions[alive_idx]
            old_logprobs = old_logprobs[alive_idx]
            returns = returns[alive_idx]
            advantages = advantages[alive_idx]
            old_values = old_values[alive_idx]

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Evaluate actions with current policy
            logprob, entropy = self.models[agent_id].evaluate_actions(obs, actions)

            # Get value predictions
            if self.use_centralized_critic:
                value = self.models[agent_id].get_value(state)
            else:
                value = self.models[agent_id].get_value(obs)

            # Policy loss with PPO clipping
            ratio = torch.exp(logprob - old_logprobs)
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(
                ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
            )
            policy_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

            # Value loss with clipping
            value_clipped = old_values + (value - old_values).clamp(
                -self.clip_coef, self.clip_coef
            )
            vf_losses1 = (value - returns) ** 2
            vf_losses2 = (value_clipped - returns) ** 2
            value_loss = 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))

            # Total loss
            loss = (
                policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy.mean()
            )

            # Optimize
            optimizer_idx = 0 if self.share_policy else agent_id
            self.optimizers[optimizer_idx].zero_grad()
            loss.backward()

            # Clip gradients
            if self.share_policy:
                nn.utils.clip_grad_norm_(
                    self.models[0].parameters(), self.max_grad_norm
                )
            else:
                nn.utils.clip_grad_norm_(
                    self.models[agent_id].parameters(), self.max_grad_norm
                )

            self.optimizers[optimizer_idx].step()

            # Compute KL divergence for monitoring
            approx_kl = 0.5 * torch.mean((old_logprobs - logprob) ** 2).item()

            # Accumulate stats
            total_stats["loss/policy"] += policy_loss.item()
            total_stats["loss/value"] += value_loss.item()
            total_stats["loss/total"] += loss.item()
            total_stats["stats/entropy"] += entropy.mean().item()
            total_stats["stats/approx_kl"] += approx_kl
            update_steps += 1

        # Average across agent updates
        if update_steps > 0:
            for key in total_stats:
                total_stats[key] /= update_steps

        return total_stats, update_steps

    @staticmethod
    def compute_gae(
        rewards, dones, values, next_values, gamma: float, gae_lambda: float
    ):
        """
        Compute Generalized Advantage Estimation (GAE).
        Args:
            rewards: [T, n_agents]
            dones: [T, n_agents]
            values: [T, n_agents]
            next_values: [n_agents]
            gamma: discount factor
            gae_lambda: GAE lambda parameter
        Returns:
            advantages: [T, n_agents]
            returns: [T, n_agents]
        """
        T = rewards.shape[0]
        n_agents = rewards.shape[1]
        advantages = np.zeros((T, n_agents), dtype=np.float32)
        lastgaelam = np.zeros(n_agents, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0 - dones[t]
                nextvalues = next_values
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values
        return advantages, returns
