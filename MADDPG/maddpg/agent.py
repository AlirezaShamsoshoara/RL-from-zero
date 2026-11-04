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
        grad_clip: float = 0.5,
        critic_hidden_sizes: List[int] = None,
        layer_norm: bool = False,
        twin_critics: bool = False,
        policy_delay: int = 1,
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
        # Global gradient-norm clip applied to both actor and critic. Disabled
        # if <= 0. MADDPG's joint-action critic can produce large gradients
        # early on when the value function is still calibrating; this bounds
        # that blast radius without meaningfully hurting the fitted regime.
        self.grad_clip = float(grad_clip)
        # Centralized critic gets more capacity by default (joint input space).
        self._critic_hidden_sizes = list(critic_hidden_sizes) if critic_hidden_sizes else list(hidden_sizes)
        self._layer_norm = bool(layer_norm)
        # Twin critics (MATD3-style): each agent holds two independent centralized
        # critics Q_i_a and Q_i_b. The Bellman target uses the MIN of the two
        # target critics, which curbs the joint-action-space over-estimation
        # that vanilla MADDPG exhibits on cooperative MPE tasks.
        self.twin_critics = bool(twin_critics)
        # Delayed policy updates: only step the actor (and update targets) every
        # `policy_delay` critic updates. TD3 standard is 2.
        self.policy_delay = max(1, int(policy_delay))
        self._update_count = 0
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

        # Per-agent networks. When `twin_critics` is True each agent holds two
        # centralized critics (Q_a, Q_b) with independent init and optimizers;
        # the target uses min(Q_a_target, Q_b_target). This is the MATD3 trick.
        self.actors: List[Actor] = []
        self.actors_target: List[Actor] = []
        self.critics: List[Critic] = []
        self.critics_target: List[Critic] = []
        self.critics_b: List[Critic] = []
        self.critics_b_target: List[Critic] = []
        self.actor_optimizers: List[Adam] = []
        self.critic_optimizers: List[Adam] = []
        self.critic_b_optimizers: List[Adam] = []

        def _make_critic() -> Critic:
            return Critic(
                total_obs_dim=self.total_obs_dim,
                total_act_dim=self.total_act_dim,
                hidden_sizes=self._critic_hidden_sizes,
                activation=activation,
                layer_norm=self._layer_norm,
            ).to(self.device)

        for i in range(n_agents):
            actor = Actor(
                obs_dim=obs_dims[i], act_dim=act_dims[i],
                hidden_sizes=hidden_sizes, activation=activation,
                action_low=self.action_lows[i], action_high=self.action_highs[i],
                layer_norm=self._layer_norm,
            ).to(self.device)
            actor_target = Actor(
                obs_dim=obs_dims[i], act_dim=act_dims[i],
                hidden_sizes=hidden_sizes, activation=activation,
                action_low=self.action_lows[i], action_high=self.action_highs[i],
                layer_norm=self._layer_norm,
            ).to(self.device)
            actor_target.load_state_dict(actor.state_dict())

            critic = _make_critic()
            critic_target = _make_critic()
            critic_target.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.actors_target.append(actor_target)
            self.critics.append(critic)
            self.critics_target.append(critic_target)
            self.actor_optimizers.append(Adam(actor.parameters(), lr=actor_lr))
            self.critic_optimizers.append(Adam(critic.parameters(), lr=critic_lr))

            if self.twin_critics:
                critic_b = _make_critic()
                critic_b_target = _make_critic()
                critic_b_target.load_state_dict(critic_b.state_dict())
                self.critics_b.append(critic_b)
                self.critics_b_target.append(critic_b_target)
                self.critic_b_optimizers.append(Adam(critic_b.parameters(), lr=critic_lr))

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

            # Cast to the env's expected dtype (PettingZoo warns on float64 inputs
            # to a float32 Box, even when values are in-bounds).
            actions.append(action.astype(np.float32, copy=False))

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
        num_actor_updates = 0

        # Whether this call performs a delayed actor + target update this step.
        do_actor_update = (self._update_count % self.policy_delay) == 0
        self._update_count += 1

        # Joint-obs and joint-action tensors are shared across every agent's
        # critic and actor updates within this batch.
        all_obs = torch.cat(obs_list, dim=-1)
        all_actions = torch.cat(actions_list, dim=-1)
        all_next_obs = torch.cat(next_obs_list, dim=-1)

        # --- Shared target next-actions from target actors ---------------------
        with torch.no_grad():
            next_actions_list = []
            for i in range(self.n_agents):
                next_action = self.actors_target[i](next_obs_list[i])
                if self.target_policy_noise > 0.0:
                    noise = torch.randn_like(next_action) * self.target_policy_noise
                    if self.target_noise_clip > 0.0:
                        noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                    next_action = next_action + noise
                next_action = next_action.clamp(self.action_lows[i], self.action_highs[i])
                next_actions_list.append(next_action)
            all_next_actions = torch.cat(next_actions_list, dim=-1)

        for agent_idx in range(self.n_agents):
            # ---------- Critic(s) update ----------
            with torch.no_grad():
                target_q_a = self.critics_target[agent_idx](all_next_obs, all_next_actions)
                if self.twin_critics:
                    target_q_b = self.critics_b_target[agent_idx](all_next_obs, all_next_actions)
                    # Min-of-two-critics on the target reduces MADDPG's
                    # positive over-estimation bias on the joint action space.
                    target_q = torch.min(target_q_a, target_q_b)
                else:
                    target_q = target_q_a
                target = rewards[agent_idx] + (1.0 - dones[agent_idx]) * self.gamma * target_q

            current_q_a = self.critics[agent_idx](all_obs, all_actions)
            critic_loss_a = F.mse_loss(current_q_a, target)
            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss_a.backward()
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.critics[agent_idx].parameters(), self.grad_clip
                )
            self.critic_optimizers[agent_idx].step()

            critic_loss_val = float(critic_loss_a.item())
            current_q_val = float(current_q_a.mean().item())

            if self.twin_critics:
                current_q_b = self.critics_b[agent_idx](all_obs, all_actions)
                critic_loss_b = F.mse_loss(current_q_b, target)
                self.critic_b_optimizers[agent_idx].zero_grad()
                critic_loss_b.backward()
                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.critics_b[agent_idx].parameters(), self.grad_clip
                    )
                self.critic_b_optimizers[agent_idx].step()
                critic_loss_val = 0.5 * (critic_loss_val + float(critic_loss_b.item()))
                current_q_val = 0.5 * (current_q_val + float(current_q_b.mean().item()))

            total_critic_loss += critic_loss_val
            total_q_value += current_q_val

            # ---------- Actor + target-net updates (delayed) ----------
            if not do_actor_update:
                continue

            # Actor loss under the CURRENT joint policy (every agent's live
            # actor). Others detached so gradients only touch this agent's
            # optimizer.
            policy_actions_list = []
            for i in range(self.n_agents):
                if i == agent_idx:
                    policy_actions_list.append(self.actors[i](obs_list[i]))
                else:
                    with torch.no_grad():
                        policy_actions_list.append(self.actors[i](obs_list[i]))
            all_policy_actions = torch.cat(policy_actions_list, dim=-1)

            actor_loss = -self.critics[agent_idx](all_obs, all_policy_actions).mean()
            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.actors[agent_idx].parameters(), self.grad_clip
                )
            self.actor_optimizers[agent_idx].step()
            total_actor_loss += float(actor_loss.item())
            num_actor_updates += 1

            # Soft-update target networks together with the actor step.
            self._soft_update(self.actors[agent_idx], self.actors_target[agent_idx])
            self._soft_update(self.critics[agent_idx], self.critics_target[agent_idx])
            if self.twin_critics:
                self._soft_update(self.critics_b[agent_idx], self.critics_b_target[agent_idx])

        self.global_step += 1

        # When the actor was skipped this step, report the previous actor loss
        # (any float; zero is a fine stand-in) rather than divide by zero.
        actor_loss_mean = (total_actor_loss / num_actor_updates) if num_actor_updates > 0 else 0.0

        return MADDPGStats(
            critic_loss=total_critic_loss / self.n_agents,
            actor_loss=actor_loss_mean,
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
            "twin_critics": self.twin_critics,
        }
        for i in range(self.n_agents):
            state[f"actor_{i}_state_dict"] = self.actors[i].state_dict()
            state[f"actor_{i}_target_state_dict"] = self.actors_target[i].state_dict()
            state[f"critic_{i}_state_dict"] = self.critics[i].state_dict()
            state[f"critic_{i}_target_state_dict"] = self.critics_target[i].state_dict()
            state[f"actor_{i}_optimizer_state_dict"] = self.actor_optimizers[i].state_dict()
            state[f"critic_{i}_optimizer_state_dict"] = self.critic_optimizers[i].state_dict()
            if self.twin_critics:
                state[f"critic_b_{i}_state_dict"] = self.critics_b[i].state_dict()
                state[f"critic_b_{i}_target_state_dict"] = self.critics_b_target[i].state_dict()
                state[f"critic_b_{i}_optimizer_state_dict"] = self.critic_b_optimizers[i].state_dict()
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

            if self.twin_critics and f"critic_b_{i}_state_dict" in state:
                self.critics_b[i].load_state_dict(state[f"critic_b_{i}_state_dict"])
                if f"critic_b_{i}_target_state_dict" in state:
                    self.critics_b_target[i].load_state_dict(state[f"critic_b_{i}_target_state_dict"])
                if f"critic_b_{i}_optimizer_state_dict" in state:
                    self.critic_b_optimizers[i].load_state_dict(state[f"critic_b_{i}_optimizer_state_dict"])
