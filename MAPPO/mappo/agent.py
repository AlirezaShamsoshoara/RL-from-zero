from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from .networks import ActorCritic


@dataclass
class Batch:
    obs: torch.Tensor
    state: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    alive_mask: torch.Tensor


class MAPPOAgent:
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
        critic_learning_rate=None,
        separate_grad_clipping: bool = False,
        clip_value_loss: bool = True,
        normalize_inputs: bool = False,
    ):
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v <= 0
            for v in (n_agents, obs_dim, act_dim, state_dim)
        ):
            raise ValueError("agent counts and dimensions must be positive integers")
        if action_type not in {"discrete", "continuous"}:
            raise ValueError("action_type must be discrete or continuous")
        if (
            not np.isfinite(lr)
            or lr <= 0
            or not np.isfinite(clip_coef)
            or not 0 < clip_coef < 1
        ):
            raise ValueError("invalid learning rate or clipping coefficient")
        if any(not np.isfinite(v) or v < 0 for v in (ent_coef, vf_coef, max_grad_norm)):
            raise ValueError(
                "loss weights and gradient clip must be finite/nonnegative"
            )
        self.n_agents, self.obs_dim, self.act_dim, self.state_dim = (
            n_agents,
            obs_dim,
            act_dim,
            state_dim,
        )
        self.device = device
        self.clip_coef, self.ent_coef, self.vf_coef = clip_coef, ent_coef, vf_coef
        self.max_grad_norm = max_grad_norm
        self.share_policy, self.use_centralized_critic = (
            share_policy,
            use_centralized_critic,
        )
        self.action_type = action_type
        self.separate_grad_clipping = separate_grad_clipping
        self.clip_value_loss = clip_value_loss
        self.normalize_inputs = normalize_inputs
        critic_lr = lr if critic_learning_rate is None else critic_learning_rate
        if not np.isfinite(critic_lr) or critic_lr <= 0:
            raise ValueError("critic learning rate must be positive")

        def make_optimizer(model):
            return (
                torch.optim.Adam(
                    [
                        {"params": model.actor.parameters(), "lr": lr},
                        {"params": model.critic.parameters(), "lr": critic_lr},
                    ]
                )
                if critic_learning_rate is not None
                else torch.optim.Adam(model.parameters(), lr=lr)
            )

        def make_model():
            return ActorCritic(
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
                normalize_inputs=normalize_inputs,
            ).to(device)

        if share_policy:
            model = make_model()
            self.models = [model] * n_agents
            self.optimizers = [make_optimizer(model)]
        else:
            self.models = [make_model() for _ in range(n_agents)]
            self.optimizers = [make_optimizer(model) for model in self.models]

    def _inputs(self, obs_list, state):
        obs = torch.as_tensor(
            np.asarray(obs_list), dtype=torch.float32, device=self.device
        )
        if obs.shape != (self.n_agents, self.obs_dim) or not torch.isfinite(obs).all():
            raise ValueError(
                "observations must be finite and match all agent dimensions"
            )
        if self.use_centralized_critic:
            if state is None:
                raise ValueError("centralized critic requires a joint state")
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            if state_t.shape != (self.state_dim,) or not torch.isfinite(state_t).all():
                raise ValueError("joint state must match the finite critic dimension")
        else:
            state_t = None
        return obs, state_t

    def _values(self, obs, state):
        if self.share_policy:
            if self.use_centralized_critic:
                return self.models[0].get_value(state).repeat(self.n_agents)
            return self.models[0].get_value(obs)
        return torch.stack(
            [
                model.get_value(state if self.use_centralized_critic else obs[i])
                for i, model in enumerate(self.models)
            ]
        )

    @torch.no_grad()
    def values(self, obs_list, state=None):
        """Critic-only bootstrap queries do not sample policy actions."""
        obs, state_t = self._inputs(obs_list, state)
        return self._values(obs, state_t).cpu().numpy()

    @torch.no_grad()
    def act(self, obs_list: List[np.ndarray], state=None, deterministic=False):
        obs, state_t = self._inputs(obs_list, state)
        if self.share_policy:
            actions, logprobs, _ = self.models[0].act(obs, deterministic=deterministic)
        else:
            outcomes = [
                model.act(obs[i], deterministic=deterministic)
                for i, model in enumerate(self.models)
            ]
            actions = torch.stack([out[0] for out in outcomes])
            logprobs = torch.stack([out[1] for out in outcomes])
        values = self._values(obs, state_t)
        return actions.cpu().numpy(), logprobs.cpu().numpy(), values.cpu().numpy()

    @torch.no_grad()
    def act_many(self, observations, deterministic=False):
        """Batched independent teams: [env, agent, observation]."""
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if (
            obs.ndim != 3
            or obs.shape[1:] != (self.n_agents, self.obs_dim)
            or not torch.isfinite(obs).all()
        ):
            raise ValueError("batched team observations have invalid shape or values")
        e = len(obs)
        if self.share_policy:
            actions, logprobs, _ = self.models[0].act(
                obs.reshape(-1, self.obs_dim), deterministic=deterministic
            )
            actions = (
                actions.reshape(e, self.n_agents, self.act_dim)
                if self.action_type == "continuous"
                else actions.reshape(e, self.n_agents)
            )
            logprobs = logprobs.reshape(e, self.n_agents)
        else:
            results = [
                model.act(obs[:, i], deterministic=deterministic)
                for i, model in enumerate(self.models)
            ]
            actions = torch.stack([r[0] for r in results], dim=1)
            logprobs = torch.stack([r[1] for r in results], dim=1)
        values = self._values_many(obs)
        return actions.cpu().numpy(), logprobs.cpu().numpy(), values.cpu().numpy()

    def _values_many(self, obs):
        state = obs.reshape(len(obs), -1)
        if self.share_policy:
            v = self.models[0].get_value(
                state if self.use_centralized_critic else obs.reshape(-1, self.obs_dim)
            )
            return (
                v[:, None].expand(-1, self.n_agents)
                if self.use_centralized_critic
                else v.reshape(len(obs), self.n_agents)
            )
        return torch.stack(
            [
                model.get_value(state if self.use_centralized_critic else obs[:, i])
                for i, model in enumerate(self.models)
            ],
            dim=1,
        )

    @torch.no_grad()
    def values_many(self, observations):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if (
            obs.ndim != 3
            or obs.shape[1:] != (self.n_agents, self.obs_dim)
            or not torch.isfinite(obs).all()
        ):
            raise ValueError("batched team observations have invalid shape or values")
        return self._values_many(obs).cpu().numpy()

    @torch.no_grad()
    def update_normalizers(self, observations, alive_masks=None):
        """Called between rollouts, never during collection or PPO epochs."""
        if not self.normalize_inputs:
            return
        obs = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        ).reshape(-1, self.n_agents, self.obs_dim)
        live = (
            torch.ones(obs.shape[:2], dtype=torch.bool, device=self.device)
            if alive_masks is None
            else torch.as_tensor(alive_masks, device=self.device)
            .reshape(obs.shape[:2])
            .bool()
        )
        joint = obs.reshape(len(obs), -1)
        if self.share_policy:
            self.models[0].obs_norm.update(obs[live])
            self.models[0].state_norm.update(
                joint if self.use_centralized_critic else obs[live]
            )
        else:
            for i, model in enumerate(self.models):
                model.obs_norm.update(obs[live[:, i], i])
                model.state_norm.update(
                    joint[live[:, i]]
                    if self.use_centralized_critic
                    else obs[live[:, i], i]
                )

    def _active_batch(self, batch):
        count = batch.obs.shape[0]
        if batch.obs.shape != (count, self.obs_dim) or batch.state.shape != (
            count,
            self.state_dim,
        ):
            raise ValueError("invalid rollout observation/state shapes")
        expected = (
            (count, self.act_dim) if self.action_type == "continuous" else (count,)
        )
        if batch.actions.shape != expected or any(
            getattr(batch, k).shape != (count,)
            for k in ("logprobs", "returns", "advantages", "values", "alive_mask")
        ):
            raise ValueError("invalid rollout action/statistic shapes")
        if not torch.all((batch.alive_mask == 0) | (batch.alive_mask == 1)):
            raise ValueError("alive masks must be binary")
        mask = batch.alive_mask > 0.5
        if not torch.any(mask):
            return None
        fields = {key: getattr(batch, key)[mask] for key in Batch.__dataclass_fields__}
        if any(not torch.isfinite(value).all() for value in fields.values()):
            raise ValueError("active rollout samples must be finite")
        if self.action_type == "discrete" and (
            torch.any(fields["actions"] < 0)
            or torch.any(fields["actions"] >= self.act_dim)
            or torch.any(fields["actions"] != fields["actions"].long())
        ):
            raise ValueError("discrete actions must be valid integer indices")
        return Batch(**fields)

    def update(self, batches: List[Batch]) -> Tuple[Dict[str, float], int]:
        if len(batches) != self.n_agents:
            raise ValueError("one rollout batch is required per agent")
        active = [self._active_batch(batch) for batch in batches]
        if self.share_policy:
            live = [batch for batch in active if batch is not None]
            groups = (
                []
                if not live
                else [
                    (
                        0,
                        Batch(
                            **{
                                field: torch.cat(
                                    [getattr(batch, field) for batch in live]
                                )
                                for field in Batch.__dataclass_fields__
                            }
                        ),
                    )
                ]
            )
        else:
            groups = [(i, batch) for i, batch in enumerate(active) if batch is not None]
        keys = (
            "loss/policy",
            "loss/value",
            "loss/total",
            "stats/entropy",
            "stats/approx_kl",
        )
        totals = dict.fromkeys(keys, 0.0)
        successful_steps = skipped_steps = 0
        for index, batch in groups:
            model, optimizer = self.models[index], self.optimizers[index]
            advantages = (batch.advantages - batch.advantages.mean()) / (
                batch.advantages.std(unbiased=False) + 1e-8
            )
            logprob, entropy = model.evaluate_actions(batch.obs, batch.actions)
            values = model.get_value(
                batch.state if self.use_centralized_critic else batch.obs
            )
            log_ratio = logprob - batch.logprobs
            if (
                not torch.isfinite(log_ratio).all()
                or not torch.isfinite(values).all()
                or not torch.isfinite(entropy).all()
                or torch.any(log_ratio.abs() > 20.0)
            ):
                # A clipped forward loss can still hide exp overflow in an
                # unselected branch (0 * inf backward). Reject before exp/Adam.
                optimizer.zero_grad(set_to_none=True)
                skipped_steps += 1
                continue
            ratio = torch.exp(log_ratio)
            policy_loss = torch.maximum(
                -advantages * ratio,
                -advantages * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef),
            ).mean()
            clipped_value = batch.values + (values - batch.values).clamp(
                -self.clip_coef, self.clip_coef
            )
            value_loss = (
                0.5
                * torch.maximum(
                    (values - batch.returns).square(),
                    (clipped_value - batch.returns).square(),
                ).mean()
                if self.clip_value_loss
                else 0.5 * (values - batch.returns).square().mean()
            )
            entropy_mean = entropy.mean()
            loss = (
                policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_mean
            )
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                skipped_steps += 1
                continue
            optimizer.zero_grad()
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                optimizer.zero_grad(set_to_none=True)
                skipped_steps += 1
                continue
            try:
                limit = self.max_grad_norm if self.max_grad_norm > 0 else float("inf")
                if self.separate_grad_clipping:
                    nn.utils.clip_grad_norm_(
                        model.actor.parameters(), limit, error_if_nonfinite=True
                    )
                    nn.utils.clip_grad_norm_(
                        model.critic.parameters(), limit, error_if_nonfinite=True
                    )
                else:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), limit, error_if_nonfinite=True
                    )
            except RuntimeError as error:
                optimizer.zero_grad(set_to_none=True)
                if (
                    "non-finite" not in str(error).lower()
                    and "nonfinite" not in str(error).lower()
                ):
                    raise
                skipped_steps += 1
                continue
            optimizer.step()
            successful_steps += 1
            current = {
                "loss/policy": float(policy_loss.item()),
                "loss/value": float(value_loss.item()),
                "loss/total": float(loss.item()),
                "stats/entropy": float(entropy_mean.item()),
                "stats/approx_kl": float(
                    (0.5 * (batch.logprobs - logprob).square()).mean().item()
                ),
            }
            for key in keys:
                totals[key] += current[key]
        if successful_steps:
            totals = {key: value / successful_steps for key, value in totals.items()}
        totals["stats/skipped_updates"] = float(skipped_steps)
        return totals, successful_steps

    @staticmethod
    def compute_gae(
        rewards,
        dones,
        values,
        next_values,
        gamma,
        gae_lambda,
        *,
        next_value_sequence=None,
        terminations=None,
    ):
        """Transition-aligned GAE; timeouts bootstrap but stop episode recursion."""
        if next_value_sequence is None:
            next_value_sequence = np.concatenate(
                (values[1:], np.asarray(next_values).reshape(1, -1))
            )
        if terminations is None:
            terminations = dones
        advantages = np.zeros_like(rewards, dtype=np.float32)
        tail = np.zeros(rewards.shape[1], dtype=np.float32)
        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + gamma * (1 - terminations[t]) * next_value_sequence[t]
                - values[t]
            )
            tail = delta + gamma * gae_lambda * (1 - dones[t]) * tail
            advantages[t] = tail
        return advantages, advantages + values
