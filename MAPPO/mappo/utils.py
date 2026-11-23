from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from gymnasium import spaces


ENV_DEFAULTS = dict(
    position_noise=1e-3,
    angle_noise=1e-3,
    forward_reward=1.0,
    terminate_reward=-100.0,
    fall_reward=-10.0,
    shared_reward=True,
    terminate_on_fall=True,
    remove_on_fall=True,
    max_cycles=500,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_action_grid(act_space: spaces.Box, bins: int) -> np.ndarray:
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ValueError("action_bins must be an integer at least two")
    low, high = np.asarray(act_space.low), np.asarray(act_space.high)
    if (
        low.ndim != 1
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or np.any(high <= low)
    ):
        raise ValueError("action grid requires finite ordered flat Box bounds")
    if bins ** len(low) > 1_000_000:
        raise ValueError("discrete action grid is too large")
    axes = [np.linspace(lo, hi, bins, dtype=np.float32) for lo, hi in zip(low, high)]
    return np.stack(
        [axis.reshape(-1) for axis in np.meshgrid(*axes, indexing="ij")], axis=1
    )


class DiscreteActionWrapper:
    def __init__(self, env, bins: int = 3):
        self.env = env
        self.possible_agents = list(env.possible_agents)
        if not self.possible_agents:
            raise ValueError("Cannot discretize actions without possible_agents")
        act_space = env.action_space(self.possible_agents[0])
        if not isinstance(act_space, spaces.Box):
            raise ValueError("action discretization requires a Box action space")
        self._action_vectors = _build_action_grid(act_space, bins)
        self._action_spaces = {
            name: spaces.Discrete(len(self._action_vectors))
            for name in self.possible_agents
        }

    def action_space(self, agent_id):
        return self._action_spaces[agent_id]

    def observation_space(self, agent_id):
        return self.env.observation_space(agent_id)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, actions):
        return self.env.step(
            {name: self._map_action(action) for name, action in actions.items()}
        )

    def close(self):
        return self.env.close()

    def _map_action(self, action):
        value = np.asarray(action)
        if value.ndim != 0 or not np.issubdtype(value.dtype, np.integer):
            raise ValueError("Expected an integer discrete action index")
        index = int(value)
        if not 0 <= index < len(self._action_vectors):
            raise ValueError("Discrete action index out of range")
        return self._action_vectors[index].copy()

    def __getattr__(self, name):
        return getattr(self.env, name)


def _multiwalker_factory(**kwargs):
    # Import the optional runtime dependency only when constructing a real env.
    from pettingzoo.sisl import multiwalker_v9

    return multiwalker_v9.parallel_env(**kwargs)


class MultiwalkerTimeLimitAdapter:
    """Repair legacy max-cycle flags without changing observations/rewards/physics."""

    def __init__(self, env):
        self.env = env
        raw = getattr(getattr(env, "aec_env", None), "unwrapped", None)
        self.engine = getattr(raw, "env", None)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, actions):
        observations, rewards, terms, truncs, infos = self.env.step(actions)
        if self.engine is not None and self.engine.frames >= self.engine.max_cycles:
            true_dones = self.engine.get_last_dones()
            terms, truncs = dict(terms), dict(truncs)
            for index, name in enumerate(self.env.possible_agents):
                if name in terms and terms[name] and not true_dones.get(index, False):
                    terms[name] = False
                    truncs[name] = True
        return observations, rewards, terms, truncs, infos

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_multiwalker_env(
    n_walkers: int = 3,
    seed: int = 42,
    render_mode: Optional[str] = None,
    discretize_actions: bool = False,
    action_bins: int = 3,
    env_kwargs: Optional[Dict] = None,
):
    kwargs = {**ENV_DEFAULTS, **(env_kwargs or {})}
    if "n_walkers" in kwargs or "render_mode" in kwargs:
        raise ValueError(
            "n_walkers and render_mode must be supplied through explicit arguments"
        )
    env = _multiwalker_factory(n_walkers=n_walkers, render_mode=render_mode, **kwargs)
    try:
        env = MultiwalkerTimeLimitAdapter(env)
        if discretize_actions:
            env = DiscreteActionWrapper(env, bins=action_bins)
        env.reset(seed=seed)
        for index, name in enumerate(env.possible_agents):
            env.action_space(name).seed(seed + index)
    except BaseException:
        env.close()
        raise
    return env


def get_space_dims(env, return_action_space: bool = False):
    agents = list(env.possible_agents)
    if not agents or len(agents) != len(set(agents)):
        raise ValueError("environment must have distinct possible agents")
    observation = env.observation_space(agents[0])
    action = env.action_space(agents[0])
    if not isinstance(observation, spaces.Box) or len(observation.shape) != 1:
        raise ValueError("MAPPO requires flat Box observations")
    if isinstance(action, spaces.Discrete):
        if action.start != 0:
            raise ValueError("MAPPO requires zero-based discrete actions")
        act_dim = action.n
    elif isinstance(action, spaces.Box) and len(action.shape) == 1:
        if (
            not np.isfinite(action.low).all()
            or not np.isfinite(action.high).all()
            or np.any(action.high <= action.low)
        ):
            raise ValueError("continuous actions require finite ordered bounds")
        act_dim = action.shape[0]
    else:
        raise ValueError("unsupported action space")
    for name in agents[1:]:
        if (
            env.observation_space(name) != observation
            or env.action_space(name) != action
        ):
            raise ValueError(
                "this MAPPO implementation requires homogeneous agent spaces"
            )
    result = (
        observation.shape[0],
        int(act_dim),
        observation.shape[0] * len(agents),
        len(agents),
    )
    return (*result, action) if return_action_space else result


def save_checkpoint(
    path: str,
    models: List[torch.nn.Module],
    optimizers: List[torch.optim.Optimizer],
    step: int,
    best_return: float,
    share_policy: bool = False,
    *,
    config=None,
):
    if not models or not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("checkpoint needs models and a nonnegative integer step")
    if not isinstance(share_policy, bool):
        raise ValueError("share_policy must be a bool")
    expected = 1 if share_policy else len(models)
    if len(optimizers) != expected:
        raise ValueError("optimizer count must match the policy sharing mode")
    if share_policy and len({id(model) for model in models}) != 1:
        raise ValueError("shared training models must reference the same module")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    data = {
        "step": step,
        "best_return": best_return,
        "share_policy": share_policy,
        "n_agents": len(models),
        "config": config,
    }
    if share_policy:
        data.update(
            model_state_dict=models[0].state_dict(),
            optimizer_state_dict=optimizers[0].state_dict(),
        )
    else:
        data.update(
            model_state_dicts=[model.state_dict() for model in models],
            optimizer_state_dicts=[opt.state_dict() for opt in optimizers],
        )
    torch.save(data, path)


def load_checkpoint(
    path: str,
    models: List[torch.nn.Module],
    optimizers: Optional[List[torch.optim.Optimizer]] = None,
):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not models or not isinstance(data, dict):
        raise ValueError("checkpoint and destination models are required")
    if data.get("n_agents", len(models)) != len(models):
        raise ValueError("checkpoint agent count does not match destination")
    step = data.get("step", 0)
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    shared = data.get("share_policy", False)
    if not isinstance(shared, bool):
        raise ValueError("checkpoint share_policy must be a bool")
    states = (
        [data["model_state_dict"]] * len(models)
        if shared
        else data["model_state_dicts"]
    )
    if len(states) != len(models):
        raise ValueError("checkpoint model count does not match destination")
    if not shared and len({id(model) for model in models}) != len(models):
        raise ValueError("separate policies cannot restore into a shared model")
    if optimizers is not None and len(optimizers) != (1 if shared else len(models)):
        raise ValueError("checkpoint optimizer count does not match destination")
    for model, state in zip(models, states):
        current = model.state_dict()
        saved_normalization = any(key.startswith("obs_norm.") for key in state)
        if bool(getattr(model, "normalize_inputs", False)) != saved_normalization:
            raise ValueError(
                "checkpoint input normalization does not match destination model"
            )
        for key in ("actor.action_scale", "actor.action_bias"):
            if key in current and (
                key not in state
                or not torch.equal(current[key].cpu(), state[key].cpu())
            ):
                raise ValueError(
                    "checkpoint continuous action bounds do not match environment"
                )
        model.load_state_dict(state)
    if optimizers is not None:
        saved = (
            [data["optimizer_state_dict"]] if shared else data["optimizer_state_dicts"]
        )
        if len(saved) != len(optimizers):
            raise ValueError("checkpoint optimizer count mismatch")
        for optimizer, state in zip(optimizers, saved):
            optimizer.load_state_dict(state)
    return data
