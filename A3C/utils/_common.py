"""Import-safe, native LunarLander helpers for trusted A3C snapshots."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def seed_int(value: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("must be between 0 and 2**32 - 1")
    return number


def validate_counts(seed: int, episodes: int, max_steps: int | None = None) -> None:
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if seed + episodes > 2**32:
        raise ValueError("seed + episodes - 1 must be less than 2**32")
    if max_steps is not None and (
        isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0
    ):
        raise ValueError("max_steps must be a positive integer")


def finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {name}")
    return result


def redact(value):
    """Keep provenance, but never copy credentials into JSON or W&B config."""
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if any(
                token in str(key).lower()
                for token in ("key", "token", "password", "secret", "credential")
            )
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path: str | Path) -> dict:
    """Read YAML without resolving auto devices or constructing training objects."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    if config.get("env_id") != "LunarLander-v3":
        raise ValueError("These native helpers support discrete LunarLander-v3 only")
    for name in (
        "gamma",
        "reward_scale",
        "reward_shaping",
        "hidden_sizes",
        "activation",
    ):
        if name not in config:
            raise ValueError(
                f"config must explicitly specify {name} for value provenance"
            )
    if isinstance(config["gamma"], bool) or isinstance(config["reward_scale"], bool):
        raise ValueError("gamma and reward_scale must be numbers, not booleans")
    gamma = finite(config["gamma"], "gamma")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if finite(config["reward_scale"], "reward_scale") <= 0:
        raise ValueError("reward_scale must be positive")
    if not isinstance(config["reward_shaping"], bool):
        raise ValueError("reward_shaping must be a boolean")
    hidden = config["hidden_sizes"]
    if not isinstance(hidden, (list, tuple)) or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in hidden
    ):
        raise ValueError("hidden_sizes must contain positive integers")
    if config["activation"] not in ("relu", "tanh", "elu", "leaky_relu"):
        raise ValueError("Unsupported activation")
    if not isinstance(config.get("env_kwargs", {}), dict):
        raise ValueError("env_kwargs must be a mapping")
    return config


def config_metadata(path: str | Path, config: dict) -> dict:
    return {
        "path": str(Path(path).resolve()),
        "config": redact(config),
        "digest_note": "Digest covers redacted parsed config, not secret-bearing YAML bytes",
        "redacted_sha256": hashlib.sha256(
            json.dumps(redact(config), sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
    }


def make_native_env(config: dict, render_mode: str | None = None):
    """Raw Gym factory: no custom reward wrapper, hint, or training time limit."""
    if config.get("env_id") != "LunarLander-v3":
        raise ValueError("Expected discrete LunarLander-v3")
    kwargs = dict(config.get("env_kwargs") or {})
    kwargs.pop("max_episode_steps", None)
    kwargs.pop("render_mode", None)
    # Fail closed on non-stock dynamics or wrapper-injection arguments.
    defaults = {
        "continuous": False,
        "gravity": -10.0,
        "enable_wind": False,
        "wind_power": 15.0,
        "turbulence_power": 1.5,
    }
    for key, value in kwargs.items():
        if key not in defaults or value != defaults[key]:
            raise ValueError(f"Native evaluation rejects non-stock env_kwargs: {key}")
    return gym.make(config["env_id"], render_mode=render_mode, **kwargs)


def validate_env(env) -> None:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("A3C requires flat Box observations")
    if env.observation_space.shape != (8,):
        raise ValueError("Native LunarLander requires eight observation features")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("A3C requires categorical discrete actions")
    if env.action_space.n != 4 or env.action_space.start != 0:
        raise ValueError("Native LunarLander requires four zero-based actions")


def _actor_class(source_root: str | Path | None):
    if source_root is None:
        from A3C.a3c.networks import ActorCritic

        return ActorCritic
    root = Path(source_root).resolve()
    path = root / "A3C" / "a3c" / "networks.py"
    if not path.is_file():
        raise ValueError(f"Missing trusted source root networks file: {path}")
    # Explicit file loading avoids the already-imported local A3C namespace.
    # This executes trusted source code. Never point it at untrusted repositories.
    root_text = str(root)
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    name = "_a3c_eval_networks_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import trusted networks: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ActorCritic


def build_agent(config: dict, env, state_dict: dict, source_root=None):
    """Return local ActorCritic without workers, SharedAdam, or caller RNG changes."""
    validate_env(env)
    actor_class = _actor_class(source_root)
    if "assign" in inspect.signature(torch.nn.Module.load_state_dict).parameters:
        # Meta-device initialization allocates no randomized CPU weights.
        with torch.device("meta"):
            model = actor_class(
                obs_dim=8,
                act_dim=4,
                hidden_sizes=config["hidden_sizes"],
                activation=config["activation"],
            )
        model.load_state_dict(state_dict, strict=True, assign=True)
        model.to("cpu")
    else:
        # PyTorch 2.0 compatibility: initialization is isolated and the caller's
        # CPU RNG state is restored. No evaluation action ever uses sampling.
        with torch.random.fork_rng(devices=[]):
            model = actor_class(
                obs_dim=8,
                act_dim=4,
                hidden_sizes=config["hidden_sizes"],
                activation=config["activation"],
            ).to("cpu")
        model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError("Checkpoint contains non-finite model parameters")
    return model


def load_agent(config: dict, env, checkpoint: str | Path, source_root=None):
    """Read only trusted snapshots with the parent's model_state_dict payload."""
    path = Path(checkpoint)
    before = sha256(path)
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Expected A3C checkpoint containing model_state_dict")
    if sha256(path) != before:
        raise ValueError("Checkpoint changed while loading; use an immutable snapshot")
    model = build_agent(config, env, payload["model_state_dict"], source_root)
    embedded = payload.get("config") or payload.get("training_config")
    if embedded is not None:
        if not isinstance(embedded, dict):
            raise ValueError("Checkpoint config provenance must be a mapping")
        for key in (
            "env_id",
            "env_kwargs",
            "hidden_sizes",
            "activation",
            "gamma",
            "reward_scale",
            "reward_shaping",
        ):
            expected = config.get(key, {} if key == "env_kwargs" else None)
            if key in embedded and embedded[key] != expected:
                raise ValueError(f"Checkpoint/config mismatch for {key}")
    metadata = {
        "path": str(path.resolve()),
        "sha256": before,
        "size_bytes": path.stat().st_size,
        "step": payload.get("step"),
        "best_return": payload.get("best_return"),
        "training_config": redact(embedded),
        "value_provenance": "embedded config checked"
        if embedded
        else "caller-supplied config; checkpoint has no embedded config",
    }
    for key in ("step", "best_return"):
        if metadata[key] is not None:
            number = float(metadata[key])
            metadata[key] = number if math.isfinite(number) else None
    return model, metadata


def deterministic_action(model, obs) -> tuple[int, float]:
    """Categorical argmax uses observation only, never extra info or sampling."""
    tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device="cpu")
    if tensor.shape != (8,) or not torch.isfinite(tensor).all():
        raise ValueError("Expected eight finite native observation features")
    with torch.inference_mode():
        logits, value = model(tensor)
        if logits.shape != (1, 4) or value.numel() != 1:
            raise ValueError("Unexpected A3C policy/value output shape")
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite policy logits")
        return int(logits.argmax(dim=-1).item()), finite(
            value.item(), "value prediction"
        )


def native_source_metadata(env) -> dict:
    """Confirm the installed native stop predicate, otherwise report it unknown."""
    base = env.unwrapped
    cls = type(base)
    confirmed = False
    source_hash = None
    try:
        source = inspect.getsource(cls.step)
        confirmed = cls.__module__ == "gymnasium.envs.box2d.lunar_lander" and all(
            clause in source
            for clause in (
                "if self.game_over or abs(state[0]) >= 1.0:",
                "if not self.lander.awake:",
                "terminated = True",
                "reward = +100",
            )
        )
        source_hash = hashlib.sha256(source.encode()).hexdigest()
    except (OSError, TypeError, AttributeError):
        pass
    return {
        "class": f"{cls.__module__}.{cls.__name__}",
        "gymnasium_version": gym.__version__,
        "step_source_sha256": source_hash,
        "landing_predicate_confirmed": confirmed,
        "landing_definition": "true termination, not truncated/capped, game_over false, lander asleep, inside viewport; pad location is a separate diagnostic",
        "native_horizon": getattr(
            getattr(env, "spec", None), "max_episode_steps", None
        ),
        "evaluation_reward_scale": 1.0,
        "evaluation_reward_shaping": False,
    }


def native_outcome(
    env, obs, terminated: bool, truncated: bool, capped: bool, source: dict
) -> dict:
    result = {
        "native_landing": None,
        "landed_on_pad": None,
        "crash": None,
        "out_of_bounds": None,
        "game_over": None,
        "lander_awake": None,
        "timeout": bool(truncated and not terminated),
        "capped": bool(capped),
    }
    if not source["landing_predicate_confirmed"]:
        return result
    base = env.unwrapped
    lander = getattr(base, "lander", None)
    game_over = getattr(base, "game_over", None)
    awake = getattr(lander, "awake", None)
    if game_over is None or awake is None:
        return result
    out_of_bounds = abs(float(obs[0])) >= 1.0
    result.update(
        game_over=bool(game_over),
        lander_awake=bool(awake),
        crash=bool(game_over),
        out_of_bounds=out_of_bounds,
        native_landing=bool(
            terminated
            and not truncated
            and not capped
            and not game_over
            and not awake
            and not out_of_bounds
        ),
    )
    position = getattr(lander, "position", None)
    x = getattr(position, "x", None)
    left, right = getattr(base, "helipad_x1", None), getattr(base, "helipad_x2", None)
    if x is not None and left is not None and right is not None:
        result["landed_on_pad"] = bool(result["native_landing"] and left <= x <= right)
    return result


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    result = [0.0] * len(rewards)
    total = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        total = finite(rewards[index] + gamma * total, "discounted return")
        result[index] = total
    return result


def rollout_episode(
    model,
    env,
    config: dict,
    seed: int,
    episode: int = 0,
    max_steps: int | None = None,
    capture_frames: bool = False,
    source: dict | None = None,
) -> tuple[dict, list]:
    """Complete one raw episode, or explicitly label a caller-requested cap."""
    validate_counts(seed, 1, max_steps)
    source = source if source is not None else native_source_metadata(env)
    obs, _ = env.reset(seed=seed)
    rewards, values, actions, frames = [], [], [], []
    terminated = truncated = capped = False
    if capture_frames:
        frame = env.render()
        if frame is None:
            raise RuntimeError("rgb_array rendering returned no initial frame")
        frames.append(np.asarray(frame).copy())
    while not (terminated or truncated or capped):
        action, value = deterministic_action(model, obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        terminated, truncated = bool(terminated), bool(truncated)
        rewards.append(finite(reward, "native reward"))
        values.append(value)
        actions.append(action)
        capped = bool(
            max_steps is not None
            and len(rewards) >= max_steps
            and not (terminated or truncated)
        )
        if capture_frames:
            frame = env.render()
            if frame is None:
                raise RuntimeError(
                    "rgb_array rendering returned no final/transition frame"
                )
            frames.append(np.asarray(frame).copy())
    targets = discounted_returns(rewards, float(config["gamma"]))
    shaped = config["reward_shaping"]
    scale = float(config["reward_scale"])
    native_values = (
        None if shaped else [finite(value / scale, "native value") for value in values]
    )
    outcome = native_outcome(env, obs, terminated, truncated, capped, source)
    eligible = bool(terminated and not truncated and not capped and not shaped)
    native_return = finite(sum(rewards), "native episode return")
    return {
        "episode": episode,
        "seed": seed,
        "native_return": native_return,
        "episode_length": len(rewards),
        "native_discounted_return": targets[0],
        "native_rewards": rewards,
        "actions": actions,
        "training_value_predictions": values,
        "native_value_predictions": native_values,
        "native_return_targets": targets,
        "initial_training_value": values[0],
        "initial_native_value": native_values[0] if native_values is not None else None,
        "value_units": "shaped training reward units; not convertible to native"
        if shaped
        else "native discounted reward units after V / training reward_scale",
        "target_definition": "observed discounted rewards to episode boundary, no bootstrap; truncated/capped targets are censored",
        "calibration_eligible": eligible,
        "value_return_pairs": [
            {"t": t, "value": native_values[t], "return_target": targets[t]}
            for t in range(len(rewards))
        ]
        if native_values is not None
        else [],
        "terminated": terminated,
        "truncated": truncated,
        "complete_episode": not capped,
        "benchmark_ge_200": native_return >= 200.0,
        "failure": True
        if (truncated or capped)
        else (
            not outcome["native_landing"]
            if outcome["native_landing"] is not None
            else None
        ),
        "final_observation": [finite(item, "final observation") for item in obs],
        "source_frames": len(frames),
        **outcome,
    }, frames
