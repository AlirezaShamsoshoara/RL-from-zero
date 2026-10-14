"""Fixed-agent-order MAPPO construction, provenance, and parallel rollouts."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
from gymnasium import spaces

from MAPPO.mappo.agent import MAPPOAgent
from MAPPO.mappo.config import Config
from MAPPO.mappo.diagnostics import motion_summary, package_position_x, task_status
from MAPPO.mappo.utils import (
    get_space_dims,
    load_checkpoint,
    make_multiwalker_env,
    set_seed,
)


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def seed_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 0 <= result < 2**32:
        raise argparse.ArgumentTypeError("must be between 0 and 2**32 - 1")
    return result


def validate_seed_range(seed: int, episodes: int, n_agents: int = 1) -> None:
    if seed + episodes + n_agents - 2 >= 2**32:
        raise ValueError("episode seeds plus agent ranks must be less than 2**32")


def redact(value):
    """JSON-safe recursive redaction for config provenance and native infos."""
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if any(
                    part in str(key).lower()
                    for part in ("key", "token", "password", "secret", "credential")
                )
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, np.ndarray):
        return redact(value.tolist())
    if isinstance(value, np.generic):
        return redact(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def config_mismatches(
    evaluation: dict, checkpoint: dict, prefix: str = ""
) -> list[dict]:
    differences = []
    for key in sorted(set(evaluation) | set(checkpoint)):
        left, right = evaluation.get(key), checkpoint.get(key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(left, dict) and isinstance(right, dict):
            differences.extend(config_mismatches(left, right, path))
        elif left != right or (key in evaluation) != (key in checkpoint):
            differences.append(
                {
                    "path": path,
                    "evaluation": left,
                    "checkpoint": right,
                    "evaluation_present": key in evaluation,
                    "checkpoint_present": key in checkpoint,
                }
            )
    return differences


def provenance(
    cfg: Config, payload: dict | None, seed: int, episodes: int, render_mode
) -> dict:
    evaluation = redact(cfg.to_dict())
    evaluation.update(
        {
            "seed": seed,
            "episodes": episodes,
            "render_mode": render_mode,
            "device": "cpu",
        }
    )
    saved = payload.get("config") if payload is not None else None
    if saved is not None and not isinstance(saved, dict):
        raise ValueError("checkpoint config must be a mapping when present")
    saved = redact(saved) if isinstance(saved, dict) else None
    return {
        "evaluation_config": evaluation,
        "checkpoint_config": saved,
        "checkpoint_config_status": (
            "recorded"
            if saved is not None
            else "legacy_unknown"
            if payload is not None
            else "no_checkpoint"
        ),
        "mismatches": config_mismatches(evaluation, saved)
        if saved is not None
        else None,
        "note": "Evaluation seed/episode/render overrides are separate from recorded training settings.",
    }


def create_env(cfg: Config, seed: int, render_mode=None):
    return make_multiwalker_env(
        n_walkers=cfg.n_walkers,
        seed=seed,
        render_mode=render_mode,
        discretize_actions=cfg.discretize_actions,
        action_bins=cfg.action_bins,
        env_kwargs=getattr(cfg, "env_kwargs", {}),
    )


def layout(env, cfg: Config) -> tuple[list[str], int, str]:
    ids = list(env.possible_agents)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Environment must expose nonempty unique possible_agents")
    if len(ids) != cfg.n_walkers or len(ids) != cfg.n_agents:
        raise ValueError(
            "n_walkers/n_agents must match the original possible_agents count"
        )
    first_obs = env.observation_space(ids[0])
    first_action = env.action_space(ids[0])
    if not isinstance(first_obs, spaces.Box) or len(first_obs.shape) != 1:
        raise ValueError("MAPPO helpers require homogeneous flat Box observations")
    continuous = isinstance(first_action, spaces.Box)
    if continuous:
        if (
            len(first_action.shape) != 1
            or not np.isfinite(first_action.low).all()
            or not np.isfinite(first_action.high).all()
        ):
            raise ValueError("Continuous actions require finite flat Box bounds")
        if np.any(first_action.high <= first_action.low):
            raise ValueError("Continuous action upper bounds must exceed lower bounds")
    elif not isinstance(first_action, spaces.Discrete) or first_action.start != 0:
        raise ValueError("Actions must be continuous Box or zero-based Discrete")
    for name in ids:
        obs_space, action_space = env.observation_space(name), env.action_space(name)
        if not isinstance(obs_space, spaces.Box) or obs_space.shape != first_obs.shape:
            raise ValueError("Agent observation spaces are not homogeneous")
        if continuous:
            if (
                not isinstance(action_space, spaces.Box)
                or action_space.shape != first_action.shape
            ):
                raise ValueError("Agent continuous action spaces are not homogeneous")
            if not np.array_equal(
                action_space.low, first_action.low
            ) or not np.array_equal(action_space.high, first_action.high):
                raise ValueError("Agent action bounds differ")
        elif (
            not isinstance(action_space, spaces.Discrete)
            or action_space.n != first_action.n
            or action_space.start != 0
        ):
            raise ValueError("Agent discrete action spaces are not homogeneous")
    return ids, first_obs.shape[0], "continuous" if continuous else "discrete"


def build_agent(cfg: Config, env) -> MAPPOAgent:
    layout(env, cfg)
    obs_dim, act_dim, state_dim, n_agents, action_space = get_space_dims(
        env, return_action_space=True
    )
    continuous = isinstance(action_space, spaces.Box)
    return MAPPOAgent(
        n_agents=n_agents,
        obs_dim=obs_dim,
        act_dim=act_dim,
        state_dim=state_dim,
        actor_hidden_sizes=cfg.actor_hidden_sizes,
        critic_hidden_sizes=cfg.critic_hidden_sizes,
        activation=cfg.activation,
        lr=cfg.learning_rate,
        clip_coef=cfg.clip_coef,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        device="cpu",
        share_policy=cfg.share_policy,
        use_centralized_critic=cfg.use_centralized_critic,
        action_type="continuous" if continuous else "discrete",
        action_low=action_space.low if continuous else None,
        action_high=action_space.high if continuous else None,
        critic_learning_rate=cfg.critic_learning_rate,
        separate_grad_clipping=cfg.separate_grad_clipping,
        clip_value_loss=cfg.clip_value_loss,
        normalize_inputs=cfg.normalize_inputs,
    )


def load_agent(cfg: Config, env, checkpoint: Path):
    agent = build_agent(cfg, env)
    payload = load_checkpoint(str(checkpoint), agent.models, None)
    if "share_policy" in payload and bool(payload["share_policy"]) != cfg.share_policy:
        raise ValueError("Checkpoint share_policy does not match evaluation config")
    for model in agent.models:
        model.eval()
    return agent, payload


def checkpoint_metadata(path: Path, payload: dict) -> dict:
    step = payload.get("step")
    if step is not None and (
        not isinstance(step, (int, np.integer)) or isinstance(step, bool) or step < 0
    ):
        raise ValueError("checkpoint step must be a nonnegative integer")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return redact(
        {
            "path": str(path.resolve()),
            "sha256": digest.hexdigest(),
            "size_bytes": path.stat().st_size,
            "step": payload.get("step"),
            "best_return": payload.get("best_return"),
            "share_policy": payload.get("share_policy"),
            "model_count": (
                1
                if payload.get("share_policy")
                else len(payload.get("model_state_dicts", []))
            ),
        }
    )


def native_horizon(env, cfg: Config) -> int:
    # The parent factory retains the stock max_cycles=500 before applying kwargs.
    cycles = getattr(cfg, "env_kwargs", {}).get("max_cycles", 500)
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
        raise ValueError("env_kwargs.max_cycles must be a positive integer")
    return cycles


def safety_cap(cfg: Config, horizon: int | None, override: int | None) -> int:
    cap = (
        override
        if override is not None
        else getattr(cfg, "max_steps_per_episode", horizon or 1000)
    )
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError("max_steps_per_episode must be positive")
    if override is not None and horizon is not None and override > horizon:
        raise ValueError(
            "--max-steps exceeds native max_cycles; set cfg.env_kwargs.max_cycles "
            "explicitly to the intended long horizon (for example 5000). "
            "The helper never extends the environment horizon silently."
        )
    if not math.isfinite(cfg.gamma) or not 0 <= cfg.gamma <= 1:
        raise ValueError("gamma must be finite and between 0 and 1")
    return cap


def padded_observations(
    observations: dict, ids: list[str], obs_dim: int, active: list[str]
):
    padded = []
    for name in ids:
        obs = (
            np.asarray(observations[name], dtype=np.float32)
            if name in active
            else np.zeros(obs_dim, dtype=np.float32)
        )
        if obs.shape != (obs_dim,) or not np.isfinite(obs).all():
            raise ValueError(f"Non-finite or incorrectly shaped observation for {name}")
        padded.append(obs.copy())
    return padded, np.concatenate(padded)


def active_agents(
    env, observations: dict, ids: list[str], ended: set[str]
) -> list[str]:
    if set(observations) - set(ids):
        raise ValueError("Environment returned unknown observation agent IDs")
    reported = list(getattr(env, "agents", observations.keys()))
    if set(reported) - set(ids):
        raise ValueError("Environment reported unknown active agent IDs")
    return [
        name
        for name in ids
        if name in reported and name in observations and name not in ended
    ]


def package_x(env) -> float | None:
    """Read native package position for diagnostics only, never policy inputs."""
    return package_position_x(env)


def rollout_episode(
    env,
    cfg: Config,
    agent: MAPPOAgent | None,
    seed: int,
    max_steps: int,
    horizon: int | None,
    *,
    random_policy: bool = False,
    deterministic: bool = True,
    capture_frames: bool = False,
) -> dict:
    ids, obs_dim, action_type = layout(env, cfg)
    validate_seed_range(seed, 1, len(ids))
    set_seed(seed)
    observations, reset_infos = env.reset(seed=seed)
    initial_package_x = package_x(env)
    native_status = task_status(env, initial_package_x)
    package_samples = (
        [{"parallel_step": 0, "package_x": initial_package_x}]
        if initial_package_x is not None
        else []
    )
    for rank, name in enumerate(ids):
        env.action_space(name).seed(seed + rank)
    ended = set()
    active = active_agents(env, observations, ids, ended)
    if not active:
        raise ValueError("Environment reset exposed no active observed agents")
    padded, state = padded_observations(observations, ids, obs_dim, active)
    initial_values = {name: None for name in ids}
    if agent is not None:
        values = np.asarray(agent.values(padded, state=state), dtype=float).reshape(-1)
        if values.shape != (len(ids),) or not np.isfinite(values).all():
            raise ValueError(
                "Initial critic values must contain one finite value per original agent"
            )
        initial_values.update(
            {name: float(values[i]) for i, name in enumerate(ids) if name in active}
        )
    returns = {name: 0.0 for name in ids}
    discounted_returns = {name: 0.0 for name in ids}
    end_flags = {
        name: {
            "terminated": None,
            "truncated": None,
            "removed": False,
            "capped": False,
            "end_step": None,
        }
        for name in ids
    }
    alive_history = [[name in active for name in ids]]
    infos = {name: reset_infos.get(name, {}) for name in ids}
    total, discounted, discount = 0.0, 0.0, 1.0
    transitions, length, frames = 0, 0, []

    def capture():
        frame = env.render()
        if frame is None:
            raise RuntimeError("rgb_array rendering returned no frame")
        frames.append(np.asarray(frame).copy())

    for step in range(max_steps):
        if capture_frames:
            capture()
        padded, state = padded_observations(observations, ids, obs_dim, active)
        if random_policy:
            actions = {name: env.action_space(name).sample() for name in active}
        else:
            sampled, _, _ = agent.act(padded, state=state, deterministic=deterministic)
            sampled = np.asarray(sampled)
            if sampled.shape[0] != len(ids):
                raise ValueError("Policy must return actions in original agent order")
            actions = {}
            for i, name in enumerate(ids):
                if name not in active:
                    continue
                space = env.action_space(name)
                action = (
                    np.asarray(sampled[i], dtype=np.float32)
                    if action_type == "continuous"
                    else int(np.asarray(sampled[i]).item())
                )
                if not space.contains(action):
                    raise ValueError(
                        f"Policy action for {name} is outside its action space"
                    )
                actions[name] = action
        transitions += len(active)
        next_obs, rewards, terminated, truncated, step_infos = env.step(actions)
        for mapping in (next_obs, rewards, terminated, truncated, step_infos):
            if set(mapping) - set(ids):
                raise ValueError("Parallel step returned unknown agent IDs")
        step_rewards = []
        for name in ids:
            reward = float(rewards.get(name, 0.0))
            if not math.isfinite(reward):
                raise ValueError(f"Non-finite native reward for {name}")
            step_rewards.append(reward)
            returns[name] += reward
            discounted_returns[name] += discount * reward
            if name in step_infos:
                infos[name] = step_infos[name]
            if name not in ended:
                if name in terminated:
                    end_flags[name]["terminated"] = bool(terminated[name])
                if name in truncated:
                    end_flags[name]["truncated"] = bool(truncated[name])
                if terminated.get(name, False) or truncated.get(name, False):
                    ended.add(name)
                    if end_flags[name]["end_step"] is None:
                        end_flags[name]["end_step"] = step + 1
        team_reward = sum(step_rewards) / len(ids)
        total += team_reward
        discounted += discount * team_reward
        discount *= cfg.gamma
        length = step + 1
        following = active_agents(env, next_obs, ids, ended)
        for name in active:
            if name not in following:
                end_flags[name]["removed"] = True
                if end_flags[name]["end_step"] is None:
                    end_flags[name]["end_step"] = length
                ended.add(name)
        active = following
        observations = next_obs
        alive_history.append([name in active for name in ids])
        native_status = task_status(env, initial_package_x)
        x = native_status["package_x"]
        if length % 20 == 0 and x is not None:
            package_samples.append({"parallel_step": length, "package_x": x})
        if (
            not active
            or native_status["delivered"] is True
            or native_status["failure"] is True
        ):
            break
    capped = bool(
        active
        and length == max_steps
        and native_status["delivered"] is not True
        and native_status["failure"] is not True
    )
    if capped:
        for name in active:
            end_flags[name]["capped"] = True
            end_flags[name]["end_step"] = length
    if capture_frames:
        capture()
    if not all(
        math.isfinite(value)
        for value in [
            total,
            discounted,
            *returns.values(),
            *discounted_returns.values(),
        ]
    ):
        raise ValueError("Non-finite accumulated returns")
    final_package_x = package_x(env)
    package_displacement = (
        final_package_x - initial_package_x
        if initial_package_x is not None and final_package_x is not None
        else None
    )
    if package_displacement is not None and not math.isfinite(package_displacement):
        raise ValueError("Native package displacement must be finite")
    native_status = task_status(env, initial_package_x)
    if final_package_x is not None and (
        not package_samples or package_samples[-1]["parallel_step"] != length
    ):
        package_samples.append({"parallel_step": length, "package_x": final_package_x})
    motion = motion_summary(package_samples, length, package_displacement)
    any_termination = any(flags["terminated"] is True for flags in end_flags.values())
    natural_failure = native_status["failure"] is True or (
        native_status["delivered"] is not True and any_termination
    )
    timeout = any(flags["truncated"] is True for flags in end_flags.values())
    reached_horizon = bool(horizon is not None and length >= horizon)
    terminal_reason = native_status["reason"]
    if terminal_reason == "running":
        terminal_reason = (
            "time_limit" if timeout else "safety_cap" if capped else "running"
        )
    return {
        "seed": seed,
        "agent_ids": ids,
        "team_return": total,
        "discounted_team_return": discounted,
        "per_agent_returns": returns,
        "per_agent_discounted_returns": discounted_returns,
        "initial_values": initial_values,
        "initial_package_x": initial_package_x,
        "final_package_x": final_package_x,
        "package_displacement": package_displacement,
        "package_samples": package_samples,
        "goal_x": native_status["goal_x"],
        "remaining_distance": native_status["remaining_distance"],
        "progress_fraction": native_status["progress_fraction"],
        "delivered": native_status["delivered"],
        "terminal_reason": terminal_reason,
        "task_status": native_status,
        **motion,
        "length": length,
        "agent_transitions": transitions,
        "terminated": any_termination,
        "truncated": timeout,
        "capped": capped,
        "last_end_flags": end_flags,
        "alive_history": alive_history,
        "last_infos": redact(infos),
        "native_max_cycles": horizon,
        "reached_horizon": reached_horizon,
        "survived_horizon": bool(not natural_failure and (reached_horizon or timeout)),
        "success": None,
        "frames": frames,
    }
