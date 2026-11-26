from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch.optim import Adam

from TD3.td3.networks import Actor, Critic, _action_bounds, _hidden_sizes, _positive_int


def _finite_scalar(value, name: str, minimum: float = 0.0, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    value = float(value)
    if (
        not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return value


def _cpu_snapshot(value):
    """Recursively own CPU tensors, including optimizer moments, without aliases."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return copy.deepcopy(value)


@dataclass
class TD3Stats:
    critic_loss: float
    actor_loss: Optional[float]
    q1_value: float
    q2_value: float
    target_q_value: float = 0.0
    td_error: float = 0.0
    actor_updated: bool = False


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
        max_grad_norm: float = 0.0,
    ):
        for space, name in ((obs_space, "Observation"), (act_space, "Action")):
            if (
                not isinstance(space, spaces.Box)
                or len(space.shape) != 1
                or space.shape[0] <= 0
            ):
                raise ValueError(f"{name} space must be a nonempty flat Box")
            if not np.issubdtype(space.dtype, np.floating):
                raise ValueError(f"{name} Box must have floating-point dtype")
            if not np.isfinite(space.low).all() or not np.isfinite(space.high).all():
                raise ValueError(f"{name} Box bounds must be finite")
            if not (space.low < space.high).all():
                raise ValueError(f"{name} Box bounds must satisfy low < high")
        self.obs_dim = _positive_int(obs_space.shape[0], "obs_dim")
        self.act_dim = _positive_int(act_space.shape[0], "act_dim")
        sizes = _hidden_sizes(hidden_sizes)
        actor_lr = _finite_scalar(actor_lr, "actor_lr")
        critic_lr = _finite_scalar(critic_lr, "critic_lr")
        if actor_lr == 0 or critic_lr == 0:
            raise ValueError("learning rates must be positive")
        self.gamma = _finite_scalar(gamma, "gamma", maximum=1.0)
        self.tau = _finite_scalar(tau, "tau", maximum=1.0)
        self.policy_delay = _positive_int(policy_delay, "policy_delay")
        self.target_noise = _finite_scalar(target_noise, "target_noise")
        self.noise_clip = _finite_scalar(noise_clip, "noise_clip")
        self.max_grad_norm = _finite_scalar(max_grad_norm, "max_grad_norm")
        self.device = torch.device(device)
        low, high = _action_bounds(act_space.low, act_space.high, self.act_dim)
        self.action_low = low.to(self.device).clone()
        self.action_high = high.to(self.device).clone()
        self.action_low_np = low.cpu().numpy().copy()
        self.action_high_np = high.cpu().numpy().copy()

        actor_args = (self.obs_dim, self.act_dim, sizes, activation, low, high)
        critic_args = (self.obs_dim, self.act_dim, sizes, activation)
        self.actor = Actor(*actor_args).to(self.device)
        self.actor_target = Actor(*actor_args).to(self.device)
        self.q1 = Critic(*critic_args).to(self.device)
        self.q2 = Critic(*critic_args).to(self.device)
        self.q1_target = Critic(*critic_args).to(self.device)
        self.q2_target = Critic(*critic_args).to(self.device)
        for online, target in (
            (self.actor, self.actor_target),
            (self.q1, self.q1_target),
            (self.q2, self.q2_target),
        ):
            target.load_state_dict(online.state_dict())
            target.requires_grad_(False)
        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.global_step = 0

    def _tensor(
        self, value, name: str, binary: bool = False, bounded: bool = False
    ) -> torch.Tensor:
        try:
            if isinstance(value, torch.Tensor):
                raw = value.detach().to(self.device)
                valid = not raw.is_complex() and bool(torch.isfinite(raw).all())
                binary_valid = not binary or bool(((raw == 0) | (raw == 1)).all())
                bounds_valid = not bounded or bool(
                    ((raw >= self.action_low) & (raw <= self.action_high)).all()
                )
            else:
                # NumPy preserves the precision of Python lists before the
                # float32 conversion, including exact binary-mask validation.
                array = np.asarray(value)
                valid = array.dtype.kind in "biuf" and bool(np.isfinite(array).all())
                binary_valid = not binary or bool(((array == 0) | (array == 1)).all())
                bounds_valid = not bounded or bool(
                    (
                        (array >= self.action_low_np) & (array <= self.action_high_np)
                    ).all()
                )
                raw = torch.as_tensor(array, device=self.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"{name} must be a real numeric tensor or array of compatible shape"
            ) from exc
        if not valid:
            raise ValueError(f"{name} must contain finite real values")
        if not binary_valid:
            raise ValueError(f"{name} must contain only 0 or 1")
        if not bounds_valid:
            raise ValueError(f"{name} must be within the action Box bounds")
        result = raw.to(dtype=torch.float32)
        if not torch.isfinite(result).all():
            raise ValueError(f"{name} must remain finite in float32")
        return result

    def act(
        self, obs: np.ndarray, noise: float = 0.0, deterministic: bool = False
    ) -> np.ndarray:
        noise = _finite_scalar(noise, "noise")
        if not isinstance(deterministic, (bool, np.bool_)):
            raise ValueError("deterministic must be boolean")
        obs_t = self._tensor(obs, "obs")
        single = obs_t.ndim == 1
        if single:
            obs_t = obs_t.unsqueeze(0)
        if obs_t.ndim != 2 or obs_t.shape[0] == 0 or obs_t.shape[1] != self.obs_dim:
            raise ValueError(
                f"obs must have shape ({self.obs_dim},) or nonempty (N, {self.obs_dim})"
            )
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()
        if not np.isfinite(action).all():
            raise FloatingPointError("actor produced nonfinite actions")
        # Exploration is independent of target smoothing and uses physical units.
        # Neither deterministic actions nor noise=0 touch an RNG or a counter.
        if not deterministic and noise > 0:
            with np.errstate(over="ignore", invalid="ignore"):
                action = action + np.random.normal(0.0, noise, size=action.shape)
            if not np.isfinite(action).all():
                raise FloatingPointError("exploration produced nonfinite actions")
        action = np.clip(action, self.action_low_np, self.action_high_np).astype(
            np.float32
        )
        return action[0] if single else action

    def _validated_batch(self, batch):
        try:
            obs, actions, rewards, next_obs, dones = batch
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "batch must contain obs, actions, rewards, next_obs, dones"
            ) from exc
        obs = self._tensor(obs, "obs")
        actions = self._tensor(actions, "actions", bounded=True)
        rewards = self._tensor(rewards, "rewards")
        next_obs = self._tensor(next_obs, "next_obs")
        dones = self._tensor(dones, "dones", binary=True)
        if obs.ndim != 2 or obs.shape[0] == 0 or obs.shape[1] != self.obs_dim:
            raise ValueError(f"obs must have nonempty shape (N, {self.obs_dim})")
        count = obs.shape[0]
        if next_obs.shape != obs.shape or actions.shape != (count, self.act_dim):
            raise ValueError("next_obs and actions must match the observation batch")
        for value, name in ((rewards, "rewards"), (dones, "dones")):
            if value.shape not in ((count,), (count, 1)):
                raise ValueError(f"{name} must have shape (N,) or (N, 1)")
        if not ((actions >= self.action_low) & (actions <= self.action_high)).all():
            raise ValueError("actions must be within the action Box bounds")
        # Column conversion must precede the Bellman target, not just the loss.
        return (
            obs,
            actions,
            rewards.reshape(count, 1),
            next_obs,
            dones.reshape(count, 1),
        )

    @staticmethod
    def _require_finite(value: torch.Tensor, name: str) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"nonfinite {name}")

    @staticmethod
    def _mean_value(value: torch.Tensor, name: str) -> float:
        detached = value.detach()
        if detached.device.type == "mps":
            mean = (detached / detached.numel()).sum()
        else:
            mean = detached.to(dtype=torch.float64).mean()
        TD3Agent._require_finite(mean, name)
        return float(mean.item())

    def _check_gradients(self, parameters, name: str) -> None:
        norms = []
        for parameter in parameters:
            if parameter.grad is not None:
                self._require_finite(parameter.grad, f"{name} gradient")
                # Widen norm accumulation where supported, without CPU copies.
                dtype = (
                    torch.float32 if parameter.device.type == "mps" else torch.float64
                )
                norms.append(
                    torch.linalg.vector_norm(parameter.grad.detach(), dtype=dtype)
                )
        if not norms:
            raise FloatingPointError(f"{name} has no gradients")
        norm = torch.linalg.vector_norm(torch.stack(norms))
        self._require_finite(norm, f"{name} gradient norm")
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters, self.max_grad_norm, error_if_nonfinite=True
            )
            for parameter in parameters:
                if parameter.grad is not None:
                    self._require_finite(parameter.grad, f"clipped {name} gradient")

    def update(self, batch) -> TD3Stats:
        """One critic update, with delayed actor and all target updates.

        Invalid input is rejected before RNG use or live mutation. Numerical
        failures are explicit, not sanitized or rolled back. If the critic step
        succeeds and a later actor step fails, the completed critic still counts.
        No per-update CPU checkpoint cloning is performed.
        """
        obs, actions, rewards, next_obs, dones = self._validated_batch(batch)
        with torch.no_grad():
            # Target Gaussian standard deviation and clip are in physical action
            # units, never scaled by action_scale or exploration noise.
            if self.target_noise > 0 and self.noise_clip > 0:
                noise = torch.randn_like(actions) * self.target_noise
                self._require_finite(noise, "target smoothing noise")
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
            else:
                noise = torch.zeros_like(actions)
            next_actions = self.actor_target(next_obs)
            self._require_finite(next_actions, "target actor actions")
            next_actions = next_actions + noise
            self._require_finite(next_actions, "smoothed target actions")
            next_actions = next_actions.clamp(self.action_low, self.action_high)
            target_q = torch.minimum(
                self.q1_target(next_obs, next_actions),
                self.q2_target(next_obs, next_actions),
            )
            target = rewards + (1.0 - dones) * self.gamma * target_q
            self._require_finite(target_q, "target Q")
            self._require_finite(target, "Bellman target")
        current_q1 = self.q1(obs, actions)
        current_q2 = self.q2(obs, actions)
        self._require_finite(current_q1, "Q1")
        self._require_finite(current_q2, "Q2")
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        self._require_finite(critic_loss, "critic loss")
        q1_value = self._mean_value(current_q1, "mean Q1")
        q2_value = self._mean_value(current_q2, "mean Q2")
        target_q_value = self._mean_value(target_q, "mean target Q")
        errors = torch.stack(
            ((current_q1.detach() - target).abs(), (current_q2.detach() - target).abs())
        )
        td_error = self._mean_value(errors, "mean absolute TD error")
        critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._check_gradients(critic_parameters, "critic")
        self.critic_opt.step()
        for parameter in critic_parameters:
            self._require_finite(parameter, "critic parameter after optimizer step")
        self.global_step += 1

        actor_loss_value: Optional[float] = None
        actor_updated = self.global_step % self.policy_delay == 0
        if actor_updated:
            # Freeze weights, not the computation: Q1 still differentiates its
            # action input. Restore even preexisting False flags on all exits.
            flags = [parameter.requires_grad for parameter in critic_parameters]
            self.critic_opt.zero_grad(set_to_none=True)
            try:
                for parameter in critic_parameters:
                    parameter.requires_grad_(False)
                actor_loss = -self.q1(obs, self.actor(obs)).mean()
                self._require_finite(actor_loss, "actor loss")
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_parameters = list(self.actor.parameters())
                self._check_gradients(actor_parameters, "actor")
                self.actor_opt.step()
                for parameter in actor_parameters:
                    self._require_finite(
                        parameter, "actor parameter after optimizer step"
                    )
                actor_loss_value = float(actor_loss.item())
            finally:
                for parameter, flag in zip(critic_parameters, flags):
                    parameter.requires_grad_(flag)
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)

        # target_q_value is the smoothed twin-min bootstrap Q, before reward/mask.
        # td_error is mean absolute error across both pre-update critics.
        return TD3Stats(
            critic_loss=float(critic_loss.item()),
            actor_loss=actor_loss_value,
            q1_value=q1_value,
            q2_value=q2_value,
            target_q_value=target_q_value,
            td_error=td_error,
            actor_updated=actor_updated,
        )

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module):
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters()
            ):
                target_parameter.copy_(
                    (1.0 - self.tau) * target_parameter + self.tau * source_parameter
                )
                self._require_finite(
                    target_parameter, "target parameter after soft update"
                )

    def state_dict(self) -> Dict[str, Any]:
        """Immutable owned CPU snapshot, not an exact replay/environment/RNG resume."""
        return _cpu_snapshot(
            {
                "actor_state_dict": self.actor.state_dict(),
                "actor_target_state_dict": self.actor_target.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "q1_target_state_dict": self.q1_target.state_dict(),
                "q2_target_state_dict": self.q2_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_opt.state_dict(),
                "critic_optimizer_state_dict": self.critic_opt.state_dict(),
                "global_step": self.global_step,
                "policy_delay": self.policy_delay,
            }
        )

    def _validate_model_state(self, model, values, name: str):
        if not isinstance(values, Mapping):
            raise ValueError(f"{name} must be a state mapping")
        expected = model.state_dict()
        if set(values) != set(expected):
            raise ValueError(f"{name} has missing or unexpected tensor keys")
        for key, reference in expected.items():
            value = values[key]
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
                raise ValueError(f"{name}.{key} must be a dense tensor")
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError(f"{name}.{key} has incompatible shape or dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name}.{key} must be finite")
            if key in ("action_scale", "action_bias"):
                low, high = (
                    self.action_low.detach().cpu(),
                    self.action_high.detach().cpu(),
                )
                bound_reference = (
                    high / 2.0 - low / 2.0
                    if key == "action_scale"
                    else high / 2.0 + low / 2.0
                )
                if not torch.equal(value, bound_reference):
                    raise ValueError(
                        f"{name}.{key} does not match this agent's action bounds"
                    )

    def _validate_optimizer_state(self, optimizer, values, name: str):
        if not isinstance(values, Mapping) or set(values) != {"state", "param_groups"}:
            raise ValueError(f"{name} must contain state and param_groups")
        groups, slots = values["param_groups"], values["state"]
        if (
            not isinstance(groups, list)
            or len(groups) != len(optimizer.param_groups)
            or not isinstance(slots, Mapping)
        ):
            raise ValueError(f"{name} has incompatible optimizer groups")
        seen = set()
        parameters_by_id = {}
        amsgrad_by_id = {}
        for group, live in zip(groups, optimizer.param_groups):
            if not isinstance(group, Mapping) or not isinstance(
                group.get("params"), list
            ):
                raise ValueError(f"{name} has an invalid parameter group")
            ids = group["params"]
            if len(ids) != len(live["params"]):
                raise ValueError(f"{name} has incompatible parameter group sizes")
            allowed = set(live) | {"initial_lr", "param_names"}
            if set(group) - allowed:
                raise ValueError(f"{name} has unsupported parameter group settings")
            if not {"lr", "betas", "eps", "weight_decay", "amsgrad"}.issubset(group):
                raise ValueError(f"{name} has missing Adam group settings")
            for key in ("lr", "eps", "weight_decay", "initial_lr"):
                if key in group:
                    _finite_scalar(group[key], f"{name}.{key}")
            betas = group["betas"]
            if not isinstance(betas, (tuple, list)) or len(betas) != 2:
                raise ValueError(f"{name}.betas must contain two values")
            for beta in betas:
                if _finite_scalar(beta, f"{name}.beta", maximum=1.0) == 1:
                    raise ValueError(f"{name}.beta must be less than 1")
            for key in (
                "amsgrad",
                "maximize",
                "capturable",
                "differentiable",
                "decoupled_weight_decay",
            ):
                if key in group and not isinstance(group[key], bool):
                    raise ValueError(f"{name}.{key} must be boolean")
            for key in ("foreach", "fused"):
                if (
                    key in group
                    and group[key] is not None
                    and not isinstance(group[key], bool)
                ):
                    raise ValueError(f"{name}.{key} must be boolean or None")
            if group.get("differentiable", False):
                raise ValueError(f"{name} differentiable Adam is unsupported")
            if group.get("fused") and group.get("foreach"):
                raise ValueError(f"{name} cannot enable both fused and foreach")
            if group.get("capturable") and self.device.type not in {
                "cuda",
                "xpu",
                "hpu",
                "privateuseone",
                "xla",
            }:
                raise ValueError(
                    f"{name} capturable Adam is unsupported on {self.device.type}"
                )
            if "param_names" in group and (
                not isinstance(group["param_names"], list)
                or len(group["param_names"]) != len(ids)
                or not all(isinstance(item, str) for item in group["param_names"])
            ):
                raise ValueError(f"{name} has invalid param_names")
            for identifier, parameter in zip(ids, live["params"]):
                if (
                    isinstance(identifier, bool)
                    or not isinstance(identifier, Integral)
                    or identifier < 0
                    or identifier in seen
                ):
                    raise ValueError(
                        f"{name} parameter identifiers must be unique nonnegative integers"
                    )
                seen.add(identifier)
                parameters_by_id[identifier] = parameter
                amsgrad_by_id[identifier] = group["amsgrad"]
        if any(
            isinstance(key, bool) or not isinstance(key, Integral) or key not in seen
            for key in slots
        ):
            raise ValueError(f"{name} has state for unknown parameters")
        for identifier, slot in slots.items():
            if not isinstance(slot, Mapping):
                raise ValueError(f"{name} Adam slot must be a mapping")
            # Missing/empty state is a legitimate uninitialized parameter.
            if not slot:
                continue
            required = {"step", "exp_avg", "exp_avg_sq"}
            if amsgrad_by_id[identifier]:
                required.add("max_exp_avg_sq")
            if set(slot) != required:
                raise ValueError(
                    f"{name} has missing or unexpected initialized Adam slots"
                )
            step = slot["step"]
            if isinstance(step, torch.Tensor):
                if (
                    step.ndim != 0
                    or step.is_complex()
                    or step.dtype == torch.bool
                    or not torch.isfinite(step).all()
                ):
                    raise ValueError(f"{name} Adam step must be a finite scalar")
                step = step.item()
            step = _finite_scalar(step, f"{name}.step")
            if not step.is_integer():
                raise ValueError(f"{name} Adam step must be an integer count")
            parameter = parameters_by_id[identifier]
            for key in required - {"step"}:
                moment = slot[key]
                if (
                    not isinstance(moment, torch.Tensor)
                    or moment.layout != torch.strided
                    or moment.shape != parameter.shape
                    or moment.dtype != parameter.dtype
                    or not torch.isfinite(moment).all()
                ):
                    raise ValueError(
                        f"{name}.{key} must be a finite matching moment tensor"
                    )
                if key != "exp_avg" and not (moment >= 0).all():
                    raise ValueError(f"{name}.{key} must be nonnegative")
            if (
                amsgrad_by_id[identifier]
                and not (slot["max_exp_avg_sq"] >= slot["exp_avg_sq"]).all()
            ):
                raise ValueError(f"{name} AMSGrad maximum must dominate exp_avg_sq")

    def load_state_dict(self, state: Dict[str, Any]):
        """Validate the complete checkpoint before copying into live parameters.

        actor/q1/q2 are required. Legacy missing target keys copy their online
        networks; present targets are never reset. Missing optimizer keys leave
        the current optimizer untouched. Legacy missing counters use step 0 and
        the current policy delay. Present policy_delay restores update cadence.
        This does not restore replay, environments, or any RNG state.
        """
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint must be a mapping")
        models = {
            "actor_state_dict": self.actor,
            "q1_state_dict": self.q1,
            "q2_state_dict": self.q2,
            "actor_target_state_dict": self.actor_target,
            "q1_target_state_dict": self.q1_target,
            "q2_target_state_dict": self.q2_target,
        }
        for key in ("actor_state_dict", "q1_state_dict", "q2_state_dict"):
            if key not in state:
                raise ValueError(f"checkpoint requires {key}")
        # Checkpoint loading is the cold path: make one owned staging copy, not
        # aliases to the caller's state. The update hot path never calls this.
        staged = _cpu_snapshot(state)
        for key, model in models.items():
            if key not in staged:
                staged[key] = _cpu_snapshot(staged[key.replace("_target", "")])
            self._validate_model_state(model, staged[key], key)
        optimizers = {
            "actor_optimizer_state_dict": self.actor_opt,
            "critic_optimizer_state_dict": self.critic_opt,
        }
        for key, optimizer in optimizers.items():
            if key in staged:
                self._validate_optimizer_state(optimizer, staged[key], key)
        step = staged.get("global_step", 0)
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError("global_step must be a nonnegative integer critic count")
        delay = _positive_int(
            staged.get("policy_delay", self.policy_delay), "policy_delay"
        )
        for key, model in models.items():
            model.load_state_dict(staged[key], strict=True)
        for key, optimizer in optimizers.items():
            if key in staged:
                optimizer.load_state_dict(staged[key])
        for target in (self.actor_target, self.q1_target, self.q2_target):
            target.requires_grad_(False)
        self.global_step = int(step)
        self.policy_delay = delay
