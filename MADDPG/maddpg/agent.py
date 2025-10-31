from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from MADDPG.maddpg.networks import Actor, Critic


@dataclass
class MADDPGStats:
    """Training statistics for MADDPG."""
    critic_loss: float
    actor_loss: float
    q_value: float


class MADDPGAgent:
    """
    Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation.

    Each agent has:
    - Its own actor network (decentralized execution)
    - Its own critic network that observes all agents (centralized training)
    - Target networks for both actor and critic

    Key features:
    - Centralized training, decentralized execution (CTDE)
    - Continuous action spaces
    - Soft target updates (Polyak averaging)
    - Experience replay
    """

    def __init__(
        self,
        n_agents: int,
        obs_dims: List[int],
        act_dims: List[int],
        action_lows: List[np.ndarray],
        action_highs: List[np.ndarray],
        hidden_sizes: List[int],
        activation: str,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        target_policy_noise: float,
        target_noise_clip: float,
        device: str,
    ):
        """
        Initialize MADDPG agent.

        Args:
            n_agents: Number of agents
            obs_dims: List of observation dimensions for each agent
            act_dims: List of action dimensions for each agent
            action_lows: List of action lower bounds for each agent
            action_highs: List of action upper bounds for each agent
            hidden_sizes: Hidden layer sizes for networks
            activation: Activation function name
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            target_policy_noise: Noise std for target policy smoothing
            target_noise_clip: Noise clipping range for target policy
            device: Device to use (cpu or cuda)
        """
        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.target_policy_noise = float(target_policy_noise)
        self.target_noise_clip = float(target_noise_clip)
        self.global_step = 0

        # Total dimensions for centralized critic
        self.total_obs_dim = sum(obs_dims)
        self.total_act_dim = sum(act_dims)

        # Convert action bounds to tensors
        self.action_lows = [
            torch.as_tensor(low, dtype=torch.float32, device=self.device)
            for low in action_lows
        ]
        self.action_highs = [
            torch.as_tensor(high, dtype=torch.float32, device=self.device)
            for high in action_highs
        ]
        self.action_lows_np = [low for low in action_lows]
        self.action_highs_np = [high for high in action_highs]

        # Create actors, critics, and target networks for each agent
        self.actors = []
        self.actors_target = []
        self.critics = []
        self.critics_target = []
        self.actor_optimizers = []
        self.critic_optimizers = []

        for i in range(n_agents):
            # Actor for agent i
            actor = Actor(
                obs_dim=obs_dims[i],
                act_dim=act_dims[i],
                hidden_sizes=hidden_sizes,
                activation=activation,
                action_low=self.action_lows[i],
                action_high=self.action_highs[i],
            ).to(self.device)
            actor_target = Actor(
                obs_dim=obs_dims[i],
                act_dim=act_dims[i],
                hidden_sizes=hidden_sizes,
                activation=activation,
                action_low=self.action_lows[i],
                action_high=self.action_highs[i],
            ).to(self.device)
            actor_target.load_state_dict(actor.state_dict())

            # Critic for agent i (centralized - sees all obs and actions)
            critic = Critic(
                total_obs_dim=self.total_obs_dim,
                total_act_dim=self.total_act_dim,
                hidden_sizes=hidden_sizes,
                activation=activation,
            ).to(self.device)
            critic_target = Critic(
                total_obs_dim=self.total_obs_dim,
                total_act_dim=self.total_act_dim,
                hidden_sizes=hidden_sizes,
                activation=activation,
            ).to(self.device)
            critic_target.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.actors_target.append(actor_target)
            self.critics.append(critic)
            self.critics_target.append(critic_target)
            self.actor_optimizers.append(Adam(actor.parameters(), lr=actor_lr))
            self.critic_optimizers.append(Adam(critic.parameters(), lr=critic_lr))

    def act(
        self,
        obs_list: List[np.ndarray],
        noise: float = 0.0,
        deterministic: bool = False
    ) -> List[np.ndarray]:
        """
        Get actions for all agents given their observations.

        Args:
            obs_list: List of observations, one per agent
            noise: Exploration noise std (Gaussian)
            deterministic: If True, no noise is added

        Returns:
            actions: List of actions, one per agent
        """
        actions = []
        for i in range(self.n_agents):
            obs = obs_list[i]
            if obs.ndim == 1:
                obs = obs[None, :]
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                action = self.actors[i](obs_t).cpu().numpy()[0]

            # Add exploration noise
            if not deterministic and noise > 0.0:
                noise_sample = np.random.normal(0.0, noise, size=action.shape)
                action = action + noise_sample
                action = np.clip(action, self.action_lows_np[i], self.action_highs_np[i])

            actions.append(action)

        return actions

    def update(self, batch) -> MADDPGStats:
        """
        Update all agents' actors and critics using a batch of transitions.

        Args:
            batch: Tuple of (obs_list, actions_list, rewards, next_obs_list, dones)
                   where each element is a list containing data for all agents

        Returns:
            stats: Training statistics averaged across all agents
        """
        obs_list, actions_list, rewards, next_obs_list, dones = batch

        total_critic_loss = 0.0
        total_actor_loss = 0.0
        total_q_value = 0.0

        # Update each agent
        for agent_idx in range(self.n_agents):
            # ========== Update Critic ==========
            with torch.no_grad():
                # Get next actions from all target actors
                next_actions_list = []
                for i in range(self.n_agents):
                    next_action = self.actors_target[i](next_obs_list[i])

                    # Target policy smoothing
                    if self.target_policy_noise > 0.0:
                        noise = torch.randn_like(next_action) * self.target_policy_noise
                        if self.target_noise_clip > 0.0:
                            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                        next_action = next_action + noise

                    next_action = next_action.clamp(self.action_lows[i], self.action_highs[i])
                    next_actions_list.append(next_action)

                # Concatenate all next observations and actions
                all_next_obs = torch.cat(next_obs_list, dim=-1)
                all_next_actions = torch.cat(next_actions_list, dim=-1)

                # Compute target Q-value
                target_q = self.critics_target[agent_idx](all_next_obs, all_next_actions)
                target = rewards[agent_idx] + (1.0 - dones[agent_idx]) * self.gamma * target_q

            # Concatenate all current observations and actions
            all_obs = torch.cat(obs_list, dim=-1)
            all_actions = torch.cat(actions_list, dim=-1)

            # Compute current Q-value
            current_q = self.critics[agent_idx](all_obs, all_actions)
            critic_loss = F.mse_loss(current_q, target)

            # Optimize critic
            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent_idx].step()

            # ========== Update Actor ==========
            # Get actions from current actor for all agents
            policy_actions_list = []
            for i in range(self.n_agents):
                if i == agent_idx:
                    # Use current actor for this agent
                    policy_action = self.actors[i](obs_list[i])
                else:
                    # Use fixed actions from replay buffer for other agents
                    policy_action = actions_list[i].detach()
                policy_actions_list.append(policy_action)

            # Concatenate all observations and policy actions
            all_policy_actions = torch.cat(policy_actions_list, dim=-1)

            # Actor loss: negative Q-value (gradient ascent on Q)
            actor_loss = -self.critics[agent_idx](all_obs, all_policy_actions).mean()

            # Optimize actor
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_idx].step()

            # Soft update target networks
            self._soft_update(self.actors[agent_idx], self.actors_target[agent_idx])
            self._soft_update(self.critics[agent_idx], self.critics_target[agent_idx])

            # Accumulate stats
            total_critic_loss += float(critic_loss.item())
            total_actor_loss += float(actor_loss.item())
            total_q_value += float(current_q.mean().item())

        self.global_step += 1

        # Average stats across all agents
        return MADDPGStats(
            critic_loss=total_critic_loss / self.n_agents,
            actor_loss=total_actor_loss / self.n_agents,
            q_value=total_q_value / self.n_agents,
        )

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module):
        """Soft update of target network parameters: θ_target = τ*θ_source + (1-τ)*θ_target"""
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.copy_(self.tau * src_p.data + (1.0 - self.tau) * tgt_p.data)

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        state = {
            "global_step": self.global_step,
            "n_agents": self.n_agents,
        }
        for i in range(self.n_agents):
            state[f"actor_{i}_state_dict"] = self.actors[i].state_dict()
            state[f"actor_{i}_target_state_dict"] = self.actors_target[i].state_dict()
            state[f"critic_{i}_state_dict"] = self.critics[i].state_dict()
            state[f"critic_{i}_target_state_dict"] = self.critics_target[i].state_dict()
            state[f"actor_{i}_optimizer_state_dict"] = self.actor_optimizers[i].state_dict()
            state[f"critic_{i}_optimizer_state_dict"] = self.critic_optimizers[i].state_dict()
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        """Load state dict from checkpoint."""
        if "global_step" in state:
            self.global_step = int(state["global_step"])

        for i in range(self.n_agents):
            if f"actor_{i}_state_dict" in state:
                self.actors[i].load_state_dict(state[f"actor_{i}_state_dict"])
            if f"actor_{i}_target_state_dict" in state:
                self.actors_target[i].load_state_dict(state[f"actor_{i}_target_state_dict"])
            else:
                self.actors_target[i].load_state_dict(self.actors[i].state_dict())

            if f"critic_{i}_state_dict" in state:
                self.critics[i].load_state_dict(state[f"critic_{i}_state_dict"])
            if f"critic_{i}_target_state_dict" in state:
                self.critics_target[i].load_state_dict(state[f"critic_{i}_target_state_dict"])
            else:
                self.critics_target[i].load_state_dict(self.critics[i].state_dict())

            if f"actor_{i}_optimizer_state_dict" in state:
                self.actor_optimizers[i].load_state_dict(state[f"actor_{i}_optimizer_state_dict"])
            if f"critic_{i}_optimizer_state_dict" in state:
                self.critic_optimizers[i].load_state_dict(state[f"critic_{i}_optimizer_state_dict"])
