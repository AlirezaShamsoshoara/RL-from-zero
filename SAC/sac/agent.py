from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch.optim import Adam

from SAC.sac.networks import (
    GaussianPolicy,
    QNetwork,
    _finite_output,
    _hidden_sizes,
    _positive_integer,
    get_activation,
)


def _finite_number(value, name: str, minimum=None, maximum=None, strict=False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (result < minimum or (strict and result == minimum)):
        raise ValueError(
            f"{name} must be {'greater than' if strict else 'at least'} {minimum}"
        )
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _clone_state(value, cpu: bool = True):
    """Clone recursively, including optimizer moments, without storage aliases."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if cpu:
            tensor = tensor.cpu()
        return tensor.clone()
    if isinstance(value, Mapping):
        return {key: _clone_state(item, cpu=cpu) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state(item, cpu=cpu) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item, cpu=cpu) for item in value)
    return copy.deepcopy(value)


def _finite_optimizer_state(value) -> None:
    if isinstance(value, torch.Tensor):
        _finite_output(value, "optimizer state")
    elif isinstance(value, Mapping):
        for item in value.values():
            _finite_optimizer_state(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_optimizer_state(item)
    elif isinstance(value, Real) and not math.isfinite(float(value)):
        raise FloatingPointError("Nonfinite optimizer state")


@dataclass
class SACStats:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha_value: float
    log_prob: float
    q1_mean: float = 0.0
    q2_mean: float = 0.0
    target_q_mean: float = 0.0
    entropy: float = 0.0
    td_error: float = 0.0


class SACAgent:
    """Twin-critic SAC with physical-action log densities.

    global_step counts completed updates, not environment interactions. Updates
    are sequential: a later failing phase does not undo a completed critic step.
    """

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
        max_grad_norm: float = 0.0,
    ):
        if not isinstance(obs_space, spaces.Box) or not isinstance(
            act_space, spaces.Box
        ):
            raise ValueError("Observation and action spaces must be Box")
        if len(obs_space.shape) != 1 or len(act_space.shape) != 1:
            raise ValueError("Only flat observation and action spaces are supported")
        self.obs_dim = _positive_integer(obs_space.shape[0], "obs_dim")
        self.act_dim = _positive_integer(act_space.shape[0], "act_dim")
        sizes = _hidden_sizes(hidden_sizes)
        get_activation(activation)
        actor_lr = _finite_number(actor_lr, "actor_lr", minimum=0, strict=True)
        critic_lr = _finite_number(critic_lr, "critic_lr", minimum=0, strict=True)
        alpha_lr = _finite_number(alpha_lr, "alpha_lr", minimum=0, strict=True)
        self.gamma = _finite_number(gamma, "gamma", minimum=0, maximum=1)
        self.tau = _finite_number(tau, "tau", minimum=0, maximum=1)
        entropy_scale = _finite_number(
            target_entropy_scale, "target_entropy_scale", minimum=0, strict=True
        )
        self.max_grad_norm = _finite_number(max_grad_norm, "max_grad_norm", minimum=0)
        self.device = torch.device(device)
        action_low = torch.as_tensor(act_space.low, dtype=torch.float32)
        action_high = torch.as_tensor(act_space.high, dtype=torch.float32)

        self.actor = GaussianPolicy(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=sizes,
            activation=activation,
            action_low=action_low,
            action_high=action_high,
        ).to(self.device)
        self.q1 = QNetwork(self.obs_dim, self.act_dim, sizes, activation).to(
            self.device
        )
        self.q2 = QNetwork(self.obs_dim, self.act_dim, sizes, activation).to(
            self.device
        )
        self.q1_target = QNetwork(self.obs_dim, self.act_dim, sizes, activation).to(
            self.device
        )
        self.q2_target = QNetwork(self.obs_dim, self.act_dim, sizes, activation).to(
            self.device
        )
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.requires_grad_(False)
        self.q2_target.requires_grad_(False)

        self.actor_opt = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )
        self.log_alpha = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.float32, device=self.device)
        )
        self.alpha_opt = Adam([self.log_alpha], lr=alpha_lr)
        # Preserve the original objective and convention. log_prob includes the
        # affine action scale, so this target is in physical action density units.
        # Do not silently shift it to a normalized-action entropy target.
        self.target_entropy = -float(self.act_dim) * entropy_scale
        if (
            not math.isfinite(self.target_entropy)
            or abs(self.target_entropy) > torch.finfo(torch.float32).max
        ):
            raise ValueError("target entropy must be finite in float32")
        self.global_step = 0

    @property
    def alpha(self) -> torch.Tensor:
        _finite_output(self.log_alpha, "log_alpha")
        value = self.log_alpha.exp()
        _finite_output(value, "alpha")
        if not (value > 0).all():
            raise FloatingPointError("alpha must be positive")
        return value

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        try:
            obs_t = torch.as_tensor(obs, device=self.device)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError(
                "obs must be a finite real observation vector or batch"
            ) from exc
        if obs_t.is_complex() or obs_t.dtype == torch.bool:
            raise ValueError("obs must contain real numeric values")
        single = obs_t.ndim == 1
        if (
            obs_t.ndim not in (1, 2)
            or obs_t.shape[-1] != self.obs_dim
            or obs_t.numel() == 0
        ):
            raise ValueError("obs must be a nonempty observation vector or batch")
        obs_t = obs_t.detach().to(dtype=torch.float32)
        if not torch.isfinite(obs_t).all():
            raise ValueError("obs must be finite in float32")
        with torch.no_grad():
            action = (
                self.actor.deterministic(obs_t)
                if deterministic
                else self._sample(obs_t)[0]
            )
            expected = (self.act_dim,) if single else (obs_t.shape[0], self.act_dim)
            self._checked_output(action, expected, "action")
        return action.detach().cpu().numpy().copy()

    def _validate_batch(self, batch):
        if not isinstance(batch, (tuple, list)) or len(batch) != 5:
            raise ValueError(
                "batch must contain obs, actions, rewards, next_obs, dones"
            )
        obs, actions, rewards, next_obs, dones = batch
        for name, value in zip(
            ("obs", "actions", "rewards", "next_obs", "dones"), batch
        ):
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"batch {name} must be a tensor")
            if not value.is_floating_point() and not (
                name == "dones" and value.dtype == torch.bool
            ):
                raise ValueError(
                    f"batch {name} must contain real floating-point values"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"batch {name} must be finite")
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim or obs.shape[0] == 0:
            raise ValueError("batch obs must have shape (N, obs_dim) with N > 0")
        count = obs.shape[0]
        if actions.shape != (count, self.act_dim) or next_obs.shape != (
            count,
            self.obs_dim,
        ):
            raise ValueError("batch action/next-observation shapes do not match obs")
        if rewards.shape not in ((count,), (count, 1)) or dones.shape not in (
            (count,),
            (count, 1),
        ):
            raise ValueError("batch rewards and dones must have shape (N,) or (N, 1)")
        if not ((dones >= 0) & (dones <= 1)).all():
            raise ValueError("batch dones must be in [0, 1]")
        low = self.actor.action_bias - self.actor.action_scale
        high = self.actor.action_bias + self.actor.action_scale
        # Compare before float32 conversion so rounding cannot conceal violations.
        action_values = actions.detach().to(device=self.device, dtype=torch.float64)
        if not (
            (action_values >= low.double()) & (action_values <= high.double())
        ).all():
            raise ValueError("batch actions are outside the physical action bounds")
        converted = tuple(
            value.detach().to(self.device, dtype=torch.float32) for value in batch
        )
        if any(not torch.isfinite(value).all() for value in converted):
            raise ValueError("batch values must be finite in float32")
        return (
            converted[0],
            converted[1],
            converted[2].reshape(count, 1),
            converted[3],
            converted[4].reshape(count, 1),
        )

    def _checked_output(self, value, shape, name: str):
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or value.shape != shape
        ):
            raise ValueError(f"{name} must be a floating-point tensor of shape {shape}")
        _finite_output(value, name)
        return value

    def _sample(self, obs):
        values = self.actor.sample(obs)
        if not isinstance(values, (tuple, list)) or len(values) != 3:
            raise ValueError("Policy sample must return action, log_prob, mean_action")
        action_shape = (*obs.shape[:-1], self.act_dim)
        log_prob_shape = (*obs.shape[:-1], 1)
        return (
            self._checked_output(values[0], action_shape, "sampled action"),
            self._checked_output(values[1], log_prob_shape, "policy log probability"),
            self._checked_output(values[2], action_shape, "mean action"),
        )

    def _q_value(self, network, obs, actions):
        return self._checked_output(
            network(obs, actions), (*obs.shape[:-1], 1), "Q value"
        )

    def _optimizer_step(self, loss, optimizer, parameters, phase: str) -> None:
        """Reject invalid gradients before Adam and rollback an unsafe Adam phase."""
        parameters = list(parameters)
        _finite_output(loss, f"{phase} loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [
            parameter.grad for parameter in parameters if parameter.grad is not None
        ]
        if not gradients:
            raise FloatingPointError(f"No {phase} gradients")
        for gradient in gradients:
            _finite_output(gradient, f"{phase} gradient")
        norms = torch.stack(
            [torch.linalg.vector_norm(gradient.detach()) for gradient in gradients]
        )
        _finite_output(torch.linalg.vector_norm(norms), f"{phase} gradient norm")
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters, self.max_grad_norm, error_if_nonfinite=True
            )
            for gradient in gradients:
                _finite_output(gradient, f"clipped {phase} gradient")
        for parameter in parameters:
            _finite_output(parameter, f"{phase} parameter")
        _finite_optimizer_state(optimizer.state_dict())
        # These local device clones are only for this optimizer phase. They are
        # not whole-update transactions, and do not round-trip through the CPU.
        old_parameters = [parameter.detach().clone() for parameter in parameters]
        old_optimizer = _clone_state(optimizer.state_dict(), cpu=False)
        try:
            optimizer.step()
            for parameter in parameters:
                _finite_output(parameter, f"updated {phase} parameter")
            _finite_optimizer_state(optimizer.state_dict())
            if optimizer is self.alpha_opt:
                self.alpha  # Validate exponentiation, not just finite log_alpha.
        except Exception:
            with torch.no_grad():
                for parameter, previous in zip(parameters, old_parameters):
                    parameter.copy_(previous)
            optimizer.load_state_dict(old_optimizer)
            raise

    def update(self, batch) -> SACStats:
        obs, actions, rewards, next_obs, dones = self._validate_batch(batch)
        alpha = self.alpha.detach()
        self.alpha_opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            next_actions, next_log_prob, _ = self._sample(next_obs)
            target_q1 = self._q_value(self.q1_target, next_obs, next_actions)
            target_q2 = self._q_value(self.q2_target, next_obs, next_actions)
            target_v = torch.min(target_q1, target_q2) - alpha * next_log_prob
            # dones represents native termination, not an episode time limit.
            target_q = rewards + (1.0 - dones) * self.gamma * target_v
            _finite_output(target_q, "Bellman target")

        current_q1 = self._q_value(self.q1, obs, actions)
        current_q2 = self._q_value(self.q2, obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )
        critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
        self._optimizer_step(critic_loss, self.critic_opt, critic_parameters, "critic")
        self.critic_opt.zero_grad(set_to_none=True)

        # Freeze the online critics, not the action-to-Q computation graph.
        # Preserve caller flags, including on errors during actor backward.
        critic_flags = [parameter.requires_grad for parameter in critic_parameters]
        try:
            for parameter in critic_parameters:
                parameter.requires_grad_(False)
            new_actions, log_prob, _ = self._sample(obs)
            q_pi = torch.min(
                self._q_value(self.q1, obs, new_actions),
                self._q_value(self.q2, obs, new_actions),
            )
            actor_loss = (alpha * log_prob - q_pi).mean()
            self._optimizer_step(
                actor_loss, self.actor_opt, self.actor.parameters(), "actor"
            )
        finally:
            for parameter, flag in zip(critic_parameters, critic_flags):
                parameter.requires_grad_(flag)

        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()
        self._optimizer_step(alpha_loss, self.alpha_opt, [self.log_alpha], "alpha")
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        with torch.no_grad():
            # Use float64 reductions so finite large values cannot overflow a
            # diagnostic sum even when the actual losses remain finite.
            log_prob_mean = log_prob.double().mean()
            td_error = 0.5 * (
                (current_q1.double() - target_q.double()).abs().mean()
                + (current_q2.double() - target_q.double()).abs().mean()
            )
            stats = SACStats(
                critic_loss=float(critic_loss.item()),
                actor_loss=float(actor_loss.item()),
                alpha_loss=float(alpha_loss.item()),
                alpha_value=float(self.alpha.item()),
                log_prob=float(log_prob_mean.item()),
                q1_mean=float(current_q1.double().mean().item()),
                q2_mean=float(current_q2.double().mean().item()),
                target_q_mean=float(target_q.double().mean().item()),
                entropy=float(-log_prob_mean.item()),
                td_error=float(td_error.item()),
            )
        if not all(math.isfinite(value) for value in vars(stats).values()):
            raise FloatingPointError("Nonfinite SAC diagnostics")
        self.global_step += 1
        return stats

    @torch.no_grad()
    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module):
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.mul_(1.0 - self.tau).add_(src_p, alpha=self.tau)

    def state_dict(self) -> Dict[str, Any]:
        """Return an owned CPU snapshot, not a view into the running agent."""
        return _clone_state(
            {
                "actor_state_dict": self.actor.state_dict(),
                "q1_state_dict": self.q1.state_dict(),
                "q2_state_dict": self.q2.state_dict(),
                "q1_target_state_dict": self.q1_target.state_dict(),
                "q2_target_state_dict": self.q2_target.state_dict(),
                "actor_optimizer_state_dict": self.actor_opt.state_dict(),
                "critic_optimizer_state_dict": self.critic_opt.state_dict(),
                "alpha_optimizer_state_dict": self.alpha_opt.state_dict(),
                "log_alpha": self.log_alpha,
                "global_step": self.global_step,
            }
        )

    def _validated_model_state(self, module, value, name: str):
        expected = module.state_dict()
        if not isinstance(value, Mapping) or set(value) != set(expected):
            raise ValueError(f"{name} has missing or unexpected model keys")
        result = {}
        for key, reference in expected.items():
            tensor = value[key]
            if (
                not isinstance(tensor, torch.Tensor)
                or not tensor.is_floating_point()
                or tensor.shape != reference.shape
                or tensor.layout != torch.strided
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"{name}.{key} has invalid tensor shape or values")
            tensor = tensor.detach().to(device="cpu", dtype=reference.dtype).clone()
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name}.{key} overflows the model dtype")
            result[key] = tensor
        return result

    def _validated_optimizer_state(self, optimizer, value, name: str):
        if (
            not isinstance(value, Mapping)
            or "state" not in value
            or "param_groups" not in value
            or not isinstance(value["state"], Mapping)
            or not isinstance(value["param_groups"], (tuple, list))
            or len(value["param_groups"]) != len(optimizer.param_groups)
        ):
            raise ValueError(f"{name} must contain compatible state and param_groups")
        parameters_by_id = {}
        amsgrad_by_id = {}
        for saved, current in zip(value["param_groups"], optimizer.param_groups):
            if not isinstance(saved, Mapping) or not isinstance(
                saved.get("params"), (tuple, list)
            ):
                raise ValueError(f"{name} has invalid parameter groups")
            if len(saved["params"]) != len(current["params"]):
                raise ValueError(f"{name} parameter count does not match")
            for saved_id, parameter in zip(saved["params"], current["params"]):
                if (
                    isinstance(saved_id, bool)
                    or not isinstance(saved_id, Integral)
                    or saved_id in parameters_by_id
                ):
                    raise ValueError(f"{name} has invalid or duplicate parameter ids")
                parameters_by_id[saved_id] = parameter
                amsgrad_by_id[saved_id] = saved.get(
                    "amsgrad", current.get("amsgrad", False)
                )
            if saved.get("differentiable", False):
                raise ValueError(f"{name} differentiable Adam is not supported")
            if saved.get("capturable", False) and any(
                p.device.type == "cpu" for p in current["params"]
            ):
                raise ValueError(
                    f"{name} capturable Adam requires a supported non-CPU device"
                )
            if saved.get("foreach", False) and saved.get("fused", False):
                raise ValueError(f"{name} cannot enable both foreach and fused Adam")
            for key in ("lr", "eps", "weight_decay"):
                if key not in saved:
                    raise ValueError(f"{name} is missing {key}")
                _finite_number(
                    saved[key], f"{name}.{key}", minimum=0, strict=key != "weight_decay"
                )
            betas = saved.get("betas")
            if not isinstance(betas, (tuple, list)) or len(betas) != 2:
                raise ValueError(f"{name} must contain two betas")
            for beta in betas:
                if _finite_number(beta, f"{name}.beta", minimum=0) >= 1:
                    raise ValueError(f"{name} betas must be in [0, 1)")
            for key in ("amsgrad", "maximize", "capturable", "differentiable"):
                if key in saved and not isinstance(saved[key], bool):
                    raise ValueError(f"{name}.{key} must be bool")
            for key in ("foreach", "fused"):
                if (
                    key in saved
                    and saved[key] is not None
                    and not isinstance(saved[key], bool)
                ):
                    raise ValueError(f"{name}.{key} must be bool or None")
        if any(
            isinstance(key, bool) or not isinstance(key, Integral)
            for key in value["state"]
        ) or not set(value["state"]).issubset(parameters_by_id):
            raise ValueError(f"{name} references invalid or unknown parameter ids")
        for saved_id, slots in value["state"].items():
            if not isinstance(slots, Mapping):
                raise ValueError(f"{name} parameter state must be a mapping")
            if not slots:
                continue
            if not {"step", "exp_avg", "exp_avg_sq"}.issubset(slots):
                raise ValueError(f"{name} has incomplete Adam state")
            if set(slots) - {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
                raise ValueError(f"{name} has unexpected Adam state")
            if amsgrad_by_id[saved_id] and "max_exp_avg_sq" not in slots:
                raise ValueError(f"{name} AMSGrad state is missing max_exp_avg_sq")
            step = slots["step"]
            if isinstance(step, torch.Tensor):
                if (
                    step.shape not in ((), (1,))
                    or step.is_complex()
                    or not torch.isfinite(step).all()
                ):
                    raise ValueError(f"{name} has invalid Adam step")
                step = step.item()
            step = _finite_number(step, f"{name}.step", minimum=0)
            if not step.is_integer():
                raise ValueError(f"{name} Adam step must be an integer")
            parameter = parameters_by_id[saved_id]
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key not in slots:
                    continue
                tensor = slots[key]
                if (
                    not isinstance(tensor, torch.Tensor)
                    or not tensor.is_floating_point()
                    or tensor.shape != parameter.shape
                    or tensor.layout != torch.strided
                    or not torch.isfinite(tensor).all()
                    or not torch.isfinite(tensor.to(dtype=parameter.dtype)).all()
                ):
                    raise ValueError(f"{name}.{key} has invalid moment shape or values")
                if key != "exp_avg" and not (tensor >= 0).all():
                    raise ValueError(f"{name}.{key} must be nonnegative")
        # Exercise PyTorch's state-loader compatibility on a detached copy only.
        # No optimizer step or model forward is performed during validation.
        try:
            probe = copy.deepcopy(optimizer)
            probe.load_state_dict(_clone_state(value))
            _finite_optimizer_state(probe.state_dict())
        except (
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
            FloatingPointError,
        ) as exc:
            raise ValueError(f"{name} is not a compatible optimizer state") from exc
        return _clone_state(probe.state_dict())

    def load_state_dict(self, state: Dict[str, Any]):
        """Validate the whole checkpoint before mutation, then restore in place.

        Legacy checkpoints may omit target states (copy the corresponding online
        critic), optimizer states, log_alpha, and global_step (keep current values).
        The actor and both online critic states are required. No replay, RNG, or
        environment state is included, so this is not an exact process resume.
        """
        if not isinstance(state, Mapping):
            raise ValueError("Checkpoint state must be a mapping")
        model_names = {
            "actor_state_dict": self.actor,
            "q1_state_dict": self.q1,
            "q2_state_dict": self.q2,
            "q1_target_state_dict": self.q1_target,
            "q2_target_state_dict": self.q2_target,
        }
        validated = {}
        for name, module in model_names.items():
            fallback = name.replace("_target_state_dict", "_state_dict")
            if name not in state and fallback not in state:
                raise ValueError(f"Checkpoint is missing {name}")
            value = state[name] if name in state else state[fallback]
            validated[name] = self._validated_model_state(module, value, name)
        current_actor = self.actor.state_dict()
        for key in ("action_scale", "action_bias"):
            if not torch.equal(
                validated["actor_state_dict"][key], current_actor[key].detach().cpu()
            ):
                raise ValueError(
                    "Checkpoint action bounds are incompatible with this agent"
                )
        alpha_value = state.get("log_alpha", self.log_alpha.detach())
        if isinstance(alpha_value, torch.Tensor):
            if (
                alpha_value.shape not in ((), (1,))
                or not alpha_value.is_floating_point()
            ):
                raise ValueError("Checkpoint log_alpha must be a scalar or shape (1,)")
            if not torch.isfinite(alpha_value).all():
                raise ValueError("Checkpoint log_alpha must be finite")
            log_alpha = (
                alpha_value.detach()
                .to(device="cpu", dtype=torch.float32)
                .reshape(1)
                .clone()
            )
        else:
            log_alpha = torch.tensor(
                [_finite_number(alpha_value, "log_alpha")], dtype=torch.float32
            )
        alpha = log_alpha.exp()
        if (
            not torch.isfinite(log_alpha).all()
            or not torch.isfinite(alpha).all()
            or not (alpha > 0).all()
        ):
            raise ValueError("Checkpoint alpha must be finite and positive")
        validated["log_alpha"] = log_alpha
        step = state.get("global_step", self.global_step)
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError("Checkpoint global_step must be a nonnegative integer")
        validated["global_step"] = int(step)
        optimizers = {
            "actor_optimizer_state_dict": self.actor_opt,
            "critic_optimizer_state_dict": self.critic_opt,
            "alpha_optimizer_state_dict": self.alpha_opt,
        }
        for name, optimizer in optimizers.items():
            if name in state:
                validated[name] = self._validated_optimizer_state(
                    optimizer, state[name], name
                )

        previous = self.state_dict()
        try:
            self._restore_validated(validated, model_names, optimizers)
        except Exception:
            self._restore_validated(previous, model_names, optimizers)
            raise

    def _restore_validated(self, state, models, optimizers):
        for name, module in models.items():
            module.load_state_dict(state[name])
        for name, optimizer in optimizers.items():
            if name in state:
                optimizer.load_state_dict(_clone_state(state[name]))
        with torch.no_grad():
            # Never replace this Parameter: alpha_opt must retain its reference.
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.global_step = state["global_step"]
        for target in (self.q1_target, self.q2_target):
            target.requires_grad_(False)
            for parameter in target.parameters():
                parameter.grad = None
