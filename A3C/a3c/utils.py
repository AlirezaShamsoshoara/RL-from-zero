from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
import math
from numbers import Integral, Real
import os
import random
import tempfile
from typing import Optional

import gymnasium as gym
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NativeEpisodeReward(gym.Wrapper):
    """Record rewards before any learning-only shaping or scaling wrappers."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._episode_return = 0.0
        self._episode_length = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_return = 0.0
        self._episode_length = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        native_reward = float(reward)
        self._episode_return += native_reward
        self._episode_length += 1
        info = dict(info)
        info["native_reward"] = native_reward
        if terminated or truncated:
            info["native_episode_return"] = self._episode_return
            info["native_episode_length"] = self._episode_length
        return obs, reward, terminated, truncated, info


class AcrobotShapedReward(gym.Wrapper):
    """Add the potential-based bonus gamma * Phi(s') - Phi(s).

    Phi is the Acrobot tip height, with zero potential at true terminals.
    The policy-invariance argument requires consistent discounting and terminal
    handling. A time-limit truncation is not a terminal: its nonzero potential
    must be paired with a value bootstrap from the actual final observation.
    Shaping alone does not guarantee convergence of a function approximator.
    """

    def __init__(self, env: gym.Env, gamma: float = 0.99):
        super().__init__(env)
        self._gamma = gamma
        self._prev_height = -2.0

    @staticmethod
    def _tip_height(obs) -> float:
        """Tip height = -cos(theta_1) - cos(theta_1 + theta_2)."""
        cos_sum = obs[0] * obs[2] - obs[1] * obs[3]
        return float(-obs[0] - cos_sum)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_height = self._tip_height(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_height = 0.0 if terminated else self._tip_height(obs)
        shaping = self._gamma * new_height - self._prev_height
        self._prev_height = new_height
        return obs, reward + shaping, terminated, truncated, info


class RewardScaler(gym.RewardWrapper):
    """Scale learning rewards without changing native episode statistics."""

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self._scale = scale

    def reward(self, reward):
        return reward * self._scale


def _wrap_env(
    env: gym.Env,
    env_id: str,
    reward_shaping: bool,
    gamma: float = 0.99,
    reward_scale: float = 1.0,
) -> gym.Env:
    env = NativeEpisodeReward(env)
    if reward_shaping and "Acrobot" in env_id:
        env = AcrobotShapedReward(env, gamma=gamma)
    if reward_scale != 1.0:
        env = RewardScaler(env, scale=reward_scale)
    return env


def make_env(
    env_id: str,
    seed: int,
    render_mode: Optional[str] = None,
    reward_shaping: bool = False,
    gamma: float = 0.99,
    reward_scale: float = 1.0,
    env_kwargs: Optional[Mapping] = None,
) -> gym.Env:
    """Construct and seed the action space; the caller owns the first reset."""
    kwargs = dict(env_kwargs or {})
    if render_mode is not None or "render_mode" not in kwargs:
        kwargs["render_mode"] = render_mode
    env = gym.make(env_id, **kwargs)
    try:
        env = _wrap_env(env, env_id, reward_shaping, gamma, reward_scale)
        env.action_space.seed(seed)
        return env
    except BaseException:
        env.close()
        raise


def make_vec_env(
    env_id: str,
    num_envs: int,
    seed: int,
    reward_shaping: bool = False,
    gamma: float = 0.99,
    reward_scale: float = 1.0,
    env_kwargs: Optional[Mapping] = None,
) -> gym.vector.VectorEnv:
    """Use SAME_STEP autoreset; reset once with [seed + i for i in range(n)]."""
    from gymnasium.vector import AutoresetMode

    if num_envs < 1:
        raise ValueError("num_envs must be positive")

    # Keep references so a failed vector constructor cannot leak earlier envs.
    created_envs = []

    def _make_single(rank: int):
        def _thunk():
            env = make_env(
                env_id,
                seed + rank,
                reward_shaping=reward_shaping,
                gamma=gamma,
                reward_scale=reward_scale,
                env_kwargs=env_kwargs,
            )
            created_envs.append(env)
            return env

        return _thunk

    try:
        return gym.vector.SyncVectorEnv(
            [_make_single(i) for i in range(num_envs)],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
    except BaseException:
        for env in created_envs:
            try:
                env.close()
            except BaseException:
                pass
        raise


def _clone_state(value):
    """Detach every tensor, including optimizer moments, from shared storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        result = {key: _clone_state(item) for key, item in value.items()}
        if hasattr(value, "_metadata"):
            result = OrderedDict(result)
            result._metadata = _clone_state(value._metadata)
        return result
    if isinstance(value, list):
        return [_clone_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported checkpoint state type: {type(value).__name__}")


def _safe_metadata(value):
    """Use weights-only-compatible primitives and omit credential fields."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    elif callable(getattr(value, "to_dict", None)):
        value = value.to_dict()
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Checkpoint metadata keys must be strings")
            name = key.lower().replace("-", "_")
            if name in {
                "key",
                "api_key",
                "wandb_key",
                "wandb_api_key",
                "access_key",
                "private_key",
            } or any(
                part in name for part in ("secret", "password", "token", "credential")
            ):
                continue
            result[key] = _safe_metadata(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item) for item in value]
    if isinstance(value, np.generic):
        return _safe_metadata(value.item())
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported checkpoint metadata type: {type(value).__name__}")


def _checkpoint_scalars(step, best_return):
    if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    if best_return is None:
        return int(step), None
    if isinstance(best_return, bool) or not isinstance(best_return, Real):
        raise ValueError("Checkpoint best_return must be numeric or None")
    best_return = float(best_return)
    # Negative infinity is the legacy sentinel before any episode completes.
    if math.isnan(best_return) or best_return == math.inf:
        raise ValueError("Checkpoint best_return must be finite or negative infinity")
    return int(step), best_return


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_return: Optional[float],
    config=None,
    *,
    metadata: Optional[Mapping] = None,
    update_lock=None,
) -> None:
    """Snapshot under the update lock, then atomically replace the destination.

    Callers saving a live shared agent must pass that agent's update_lock.
    The optional lock keeps weights and optimizer moments from different
    updates out of the same snapshot; file I/O happens after releasing it.
    """
    step, best_return = _checkpoint_scalars(step, best_return)
    with update_lock if update_lock is not None else nullcontext():
        data = {
            "model_state_dict": _clone_state(model.state_dict()),
            "optimizer_state_dict": _clone_state(optimizer.state_dict()),
            "step": step,
            "best_return": best_return,
        }
        if config is not None:
            data["config"] = _safe_metadata(config)
            if not isinstance(data["config"], dict):
                raise TypeError("Checkpoint config must be a mapping or Config object")
        if metadata is not None:
            data["metadata"] = _safe_metadata(metadata)
            if not isinstance(data["metadata"], dict):
                raise TypeError("Checkpoint metadata must be a mapping")

    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(data, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _validate_finite_tensors(value) -> None:
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Checkpoint contains a nonfinite tensor")
    elif isinstance(value, Mapping):
        for item in value.values():
            _validate_finite_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_finite_tensors(item)


def _validate_optimizer_state(state_dict, optimizer) -> None:
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint optimizer state must be a mapping")
    states = state_dict.get("state")
    groups = state_dict.get("param_groups")
    if not isinstance(states, Mapping) or not isinstance(groups, list):
        raise ValueError("Checkpoint optimizer state requires state and param_groups")
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("Checkpoint optimizer parameter group count differs")
    parameters = {}
    for group, target in zip(groups, optimizer.param_groups):
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("Checkpoint optimizer group requires a parameter list")
        if len(group["params"]) != len(target["params"]):
            raise ValueError("Checkpoint optimizer parameter count differs")
        for key in ("lr", "eps", "weight_decay"):
            if key in group:
                number = group[key]
                if (
                    not isinstance(number, Real)
                    or not math.isfinite(number)
                    or number < 0
                ):
                    raise ValueError(f"Invalid optimizer {key}")
        if isinstance(optimizer, torch.optim.Adam):
            if not {"lr", "betas", "eps", "weight_decay", "amsgrad"}.issubset(group):
                raise ValueError("Checkpoint Adam group is missing required fields")
            betas = group["betas"]
            if (
                not isinstance(betas, (list, tuple))
                or len(betas) != 2
                or any(
                    not isinstance(beta, Real) or not 0 <= beta < 1 for beta in betas
                )
            ):
                raise ValueError("Invalid optimizer betas")
        for identifier, parameter in zip(group["params"], target["params"]):
            if (
                isinstance(identifier, bool)
                or not isinstance(identifier, Integral)
                or identifier in parameters
            ):
                raise ValueError("Invalid or duplicate optimizer parameter identifier")
            parameters[identifier] = parameter
    for identifier, state in states.items():
        if identifier not in parameters or not isinstance(state, Mapping):
            raise ValueError("Invalid optimizer parameter state")
        if (
            isinstance(optimizer, torch.optim.Adam)
            and state
            and not {"step", "exp_avg", "exp_avg_sq"}.issubset(state)
        ):
            raise ValueError("Checkpoint Adam state is missing required buffers")
        if "step" in state:
            step = state["step"]
            if isinstance(step, torch.Tensor):
                if step.numel() != 1:
                    raise ValueError("Optimizer step must be scalar")
                step = step.item()
            if (
                not isinstance(step, Real)
                or not math.isfinite(step)
                or step < 0
                or int(step) != step
            ):
                raise ValueError("Optimizer step must be a nonnegative integer")
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"):
            if name in state:
                buffer = state[name]
                parameter = parameters[identifier]
                if (
                    not isinstance(buffer, torch.Tensor)
                    or buffer.shape != parameter.shape
                    or buffer.dtype != parameter.dtype
                ):
                    raise ValueError(f"Invalid optimizer buffer {name}")
                if name.endswith("avg_sq") and bool((buffer < 0).any()):
                    raise ValueError(f"Optimizer buffer {name} must be nonnegative")
    _validate_finite_tensors(state_dict)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    update_lock=None,
):
    """Validate before mutation, load without pickle globals, and re-share Adam."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, Mapping) or not {
        "model_state_dict",
        "step",
        "best_return",
    }.issubset(data):
        raise ValueError("Checkpoint is missing required fields")
    data = dict(data)
    data["step"], data["best_return"] = _checkpoint_scalars(
        data["step"], data["best_return"]
    )
    for key in ("config", "metadata"):
        if key in data:
            data[key] = _safe_metadata(data[key])
            if not isinstance(data[key], dict):
                raise ValueError(f"Checkpoint {key} must be a mapping")
    _validate_finite_tensors(data)
    with update_lock if update_lock is not None else nullcontext():
        source = data["model_state_dict"]
        target = model.state_dict()
        if not isinstance(source, Mapping) or set(source) != set(target):
            raise ValueError("Checkpoint model state keys differ")
        for key, value in source.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != target[key].shape
                or value.dtype != target[key].dtype
            ):
                raise ValueError(f"Checkpoint model tensor differs: {key}")
        if optimizer is not None and "optimizer_state_dict" in data:
            _validate_optimizer_state(data["optimizer_state_dict"], optimizer)
        model.load_state_dict(source)
        if optimizer is not None and "optimizer_state_dict" in data:
            optimizer.load_state_dict(data["optimizer_state_dict"])
            if isinstance(optimizer, SharedAdam):
                optimizer.share_memory()
    return data


@torch.no_grad()
def compute_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    *,
    terminations: Optional[torch.Tensor] = None,
    next_values_sequence: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute boundary-aware n-step targets for (time,) or (time, env) rewards.

    Legacy dones are true terminals when terminations is omitted. Otherwise,
    dones mark all episode boundaries, and a nonterminal boundary must use the
    matching actual endpoint value in next_values_sequence, not an autoreset
    observation or a return from the following episode.
    """
    if rewards.ndim < 1 or not rewards.is_floating_point():
        raise ValueError("Rewards must be a floating tensor with a time dimension")
    if dones.shape != rewards.shape:
        raise ValueError("dones must have the same shape as rewards")
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("Rewards must be finite")
    if not bool(((dones == 0) | (dones == 1)).all()):
        raise ValueError("dones must contain only zero or one")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and between zero and one")
    boundaries = dones.to(device=rewards.device, dtype=torch.bool)
    if terminations is None:
        terminals = boundaries
    else:
        if terminations.shape != rewards.shape:
            raise ValueError("terminations must have the same shape as rewards")
        if not bool(((terminations == 0) | (terminations == 1)).all()):
            raise ValueError("terminations must contain only zero or one")
        terminals = terminations.to(device=rewards.device, dtype=torch.bool)
        if bool((terminals & ~boundaries).any()):
            raise ValueError("A true termination must also be an episode boundary")
    timeouts = boundaries & ~terminals
    if next_values_sequence is None:
        if bool(timeouts.any()):
            raise ValueError("Timeout returns require actual endpoint values")
        endpoints = torch.zeros_like(rewards)
    else:
        if next_values_sequence.shape != rewards.shape:
            raise ValueError("next_values_sequence must have the same shape as rewards")
        endpoints = next_values_sequence.to(device=rewards.device, dtype=rewards.dtype)
        if not bool(torch.isfinite(endpoints).all()):
            raise ValueError("Endpoint values must be finite")
    returns = torch.zeros_like(rewards)
    if len(rewards) == 0:
        return returns
    recursive_return = torch.as_tensor(
        last_value, device=rewards.device, dtype=rewards.dtype
    )
    if recursive_return.numel() == 1:
        recursive_return = recursive_return.reshape(())
    if not bool(torch.isfinite(recursive_return).all()):
        raise ValueError("last_value must be finite")
    try:
        recursive_return = torch.broadcast_to(recursive_return, rewards.shape[1:])
    except RuntimeError as exc:
        raise ValueError("last_value must broadcast to one reward timestep") from exc
    for step in reversed(range(len(rewards))):
        bootstrap = torch.where(boundaries[step], endpoints[step], recursive_return)
        bootstrap = torch.where(terminals[step], torch.zeros_like(bootstrap), bootstrap)
        recursive_return = rewards[step] + gamma * bootstrap
        returns[step] = recursive_return
    return returns


class SharedAdam(torch.optim.Adam):
    """CPU Adam with scalar shared steps and shared moment buffers.

    Updates still require the agent's process-shared update_lock. Sharing
    storage alone does not make the read/modify/write operations atomic.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.share_memory()

    def load_state_dict(self, state_dict):
        _validate_optimizer_state(state_dict, self)
        super().load_state_dict(state_dict)
        self.share_memory()

    def share_memory(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.device.type != "cpu":
                    raise ValueError("SharedAdam requires CPU parameters")
                state = self.state[parameter]
                step = state.get("step", 0)
                if isinstance(step, torch.Tensor):
                    if step.numel() != 1:
                        raise ValueError("SharedAdam step must be scalar")
                    step = (
                        step.detach().to(device="cpu", dtype=torch.float32).reshape(())
                    )
                else:
                    step = torch.tensor(step, device="cpu", dtype=torch.float32)
                state["step"] = step
                step.share_memory_()
                names = ["exp_avg", "exp_avg_sq"]
                if group.get("amsgrad", False):
                    names.append("max_exp_avg_sq")
                for name in names:
                    buffer = state.get(name)
                    if buffer is None:
                        buffer = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format
                        )
                    if (
                        buffer.shape != parameter.shape
                        or buffer.dtype != parameter.dtype
                        or buffer.device.type != "cpu"
                    ):
                        raise ValueError(f"Invalid SharedAdam buffer {name}")
                    state[name] = buffer.detach()
                    state[name].share_memory_()
