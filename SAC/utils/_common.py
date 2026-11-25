"""CPU observation-only SAC inference on raw, stock Pendulum-v1."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

NATIVE_HORIZON = 200
SOFT_Q_NOTE = (
    "Soft Q predicts entropy-augmented stochastic-policy continuing return, "
    "including bootstrap across native TimeLimit. Observed deterministic native "
    "reward-to-boundary has a different objective and finite horizon. Their "
    "difference is not a calibration error or proof of overestimation."
)


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
    for name, value in (("episodes", episodes), ("max_steps", max_steps)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
    if episodes is None:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if seed + episodes > 2**32:
        raise ValueError("seed + episodes - 1 must be less than 2**32")


def finite(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {name}")
    return result


def redact(value):
    """Export configuration, never credentials or arbitrary tensor objects."""
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
        return redact(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite configuration provenance")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(
        "Configuration provenance must contain only JSON-compatible values"
    )


def write_json(path: str | Path, data: dict) -> None:
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path: str | Path) -> dict:
    """No Config device resolution, optimizer construction, or training imports."""
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict) or config.get("env_id") != "Pendulum-v1":
        raise ValueError("These helpers require a Pendulum-v1 YAML mapping")
    for key in ("gamma", "hidden_sizes", "activation"):
        if key not in config:
            raise ValueError(f"Selected config must explicitly specify {key}")
    gamma = finite(config["gamma"], "gamma")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    hidden = config["hidden_sizes"]
    if not isinstance(hidden, (list, tuple)) or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in hidden
    ):
        raise ValueError("hidden_sizes must contain positive integers")
    if config["activation"] not in ("relu", "tanh", "elu", "leaky_relu"):
        raise ValueError("Unsupported activation")
    if finite(config.get("reward_scale", 1.0), "reward_scale") != 1.0:
        raise ValueError("Native helpers require reward_scale=1; no Q unit conversion")
    if config.get("reward_shaping", False) is not False:
        raise ValueError("Native helpers require reward_shaping=false")
    if config.get("normalize_rewards", False) is not False:
        raise ValueError("Native helpers do not accept normalized rewards")
    native_env_kwargs(config)
    return config


def config_metadata(path: str | Path, config: dict) -> dict:
    clean = redact(config)
    return {
        "path": str(Path(path).resolve()),
        "config": clean,
        "redacted_sha256": hashlib.sha256(
            json.dumps(clean, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "digest_note": "Digest covers redacted parsed config, not secret-bearing YAML bytes",
    }


def resolve_checkpoint(config: dict, checkpoint: str | Path | None) -> Path:
    if checkpoint is None:
        checkpoint = config.get("inference_model_path")
        if not checkpoint:
            raise ValueError(
                "Selected config must specify inference_model_path when --checkpoint is omitted"
            )
    return Path(checkpoint)


def protect_inputs(outputs, inputs) -> None:
    outputs = [Path(path) for path in outputs]
    inputs = [Path(path) for path in inputs if path is not None]
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Outputs must not overwrite each other")
    if set(resolved) & {path.resolve() for path in inputs}:
        raise ValueError(
            "Outputs must not overwrite config, checkpoint, or other source files"
        )
    # Existing hard links have different paths but can still overwrite inputs.
    existing = [path for path in outputs + inputs if path.exists()]
    for index, out in enumerate(outputs):
        if not out.exists():
            continue
        for other in outputs[index + 1 :] + inputs:
            if other in existing and out.samefile(other):
                raise ValueError("Output hard link aliases a source or another output")


def native_env_kwargs(config: dict) -> dict:
    """Pendulum's only supported physics constructor kwarg here is g."""
    if config.get("env_id") != "Pendulum-v1":
        raise ValueError("Expected Pendulum-v1")
    kwargs = config.get("env_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("env_kwargs must be a mapping")
    unknown = set(kwargs) - {"g", "max_episode_steps"}
    if unknown:
        raise ValueError(
            "Native Pendulum rejects unsupported env_kwargs: "
            + ", ".join(sorted(unknown))
        )
    if finite(kwargs.get("g", 10.0), "gravity") != 10.0:
        raise ValueError("Native evaluation rejects non-stock gravity; expected g=10")
    override = kwargs.get("max_episode_steps")
    if override is not None:
        if isinstance(override, bool) or not isinstance(override, int) or override <= 0:
            raise ValueError("Training max_episode_steps must be a positive integer")
    # max_episode_steps is a Gym wrapper option, not a physics constructor kwarg.
    # Disclose but never apply a training override to native evaluation.
    return {"g": 10.0}


def make_native_env(config: dict, render_mode: str | None = None):
    kwargs = native_env_kwargs(config)
    registered = gym.spec("Pendulum-v1").max_episode_steps
    if registered != NATIVE_HORIZON:
        raise ValueError(
            f"Registered Pendulum-v1 horizon is {registered}, expected 200"
        )
    env = gym.make("Pendulum-v1", render_mode=render_mode, **kwargs)
    try:
        validate_env(env)
    except Exception:
        env.close()
        raise
    return env


def validate_env(env) -> None:
    if not isinstance(
        env.observation_space, gym.spaces.Box
    ) or env.observation_space.shape != (3,):
        raise ValueError("Native Pendulum observations must have shape (3,)")
    if not isinstance(env.action_space, gym.spaces.Box) or env.action_space.shape != (
        1,
    ):
        raise ValueError("Native Pendulum actions must have shape (1,)")
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    if not np.array_equal(low, [-2.0]) or not np.array_equal(high, [2.0]):
        raise ValueError("Native Pendulum action bounds must be [-2, 2]")
    if getattr(getattr(env, "spec", None), "max_episode_steps", None) != NATIVE_HORIZON:
        raise ValueError(
            "Native Pendulum must retain its registered 200-transition TimeLimit"
        )
    defaults = {
        "g": 10.0,
        "m": 1.0,
        "l": 1.0,
        "dt": 0.05,
        "max_speed": 8.0,
        "max_torque": 2.0,
    }
    for name, expected in defaults.items():
        value = getattr(env.unwrapped, name, None)
        if value is not None and finite(value, name) != expected:
            raise ValueError(f"Non-stock Pendulum physics: {name}")


def _network_classes(source_root=None):
    """Load just networks.py, never an Agent or the source training package."""
    root = (
        Path(source_root).resolve()
        if source_root is not None
        else Path(__file__).resolve().parents[2]
    )
    path = root / "SAC" / "sac" / "networks.py"
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected regular trusted SAC networks source: {path}")
    before = sha256(path)
    name = "_sac_inference_networks_" + before[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import trusted networks: {path}")
    module = importlib.util.module_from_spec(spec)
    # Compile trusted source bytes directly: no __pycache__ is written into the
    # stock checkout. Avoid installing or replacing its training package.
    source_bytes = path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != before:
        raise ValueError("Networks source changed before import")
    with torch.random.fork_rng(devices=[]):
        exec(compile(source_bytes, str(path), "exec"), module.__dict__)
    if sha256(path) != before:
        raise ValueError(
            "Networks source changed while importing; use an immutable snapshot"
        )
    return (
        module.GaussianPolicy,
        module.QNetwork,
        {
            "path": str(path),
            "sha256": before,
            "source_root": str(root),
            "explicit_stock_source": source_root is not None,
            "trust_note": "Explicit source loading executes trusted Python, not a sandbox",
        },
    )


@dataclass
class InferenceModel:
    actor: torch.nn.Module
    q1: torch.nn.Module | None = None
    q2: torch.nn.Module | None = None
    network_source: dict | None = None


def _validate_state(state: dict, name: str) -> None:
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Expected nonempty {name} mapping")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must contain named tensors only")
        if value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError(f"Non-finite or non-CPU {name} tensor: {key}")


def build_agent(
    config: dict,
    env,
    state_dict: dict,
    source_root=None,
    q1_state_dict=None,
    q2_state_dict=None,
) -> InferenceModel:
    """Inference modules only; CPU fork_rng is compatible with PyTorch >=2.0."""
    validate_env(env)
    _validate_state(state_dict, "actor_state_dict")
    low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device="cpu")
    high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device="cpu")
    expected = {"action_scale": (high - low) / 2.0, "action_bias": (high + low) / 2.0}
    # Validate serialized scale/bias against the actual environment BEFORE any
    # load_state_dict mutates any module, rather than silently overwriting bounds.
    for name, value in expected.items():
        stored = state_dict.get(name)
        if (
            stored is None
            or stored.shape != value.shape
            or not torch.equal(stored, value)
        ):
            raise ValueError(
                f"Checkpoint {name} does not match native environment bounds"
            )
    for name, state in (
        ("q1_state_dict", q1_state_dict),
        ("q2_state_dict", q2_state_dict),
    ):
        if state is not None:
            _validate_state(state, name)
    actor_class, q_class, source = _network_classes(source_root)
    # Isolate source constructors as well as standard nn.Linear initialization.
    # No CUDA initialization or global seeding; caller CPU RNG is restored.
    with torch.random.fork_rng(devices=[]):
        actor = actor_class(
            obs_dim=3,
            act_dim=1,
            hidden_sizes=config["hidden_sizes"],
            activation=config["activation"],
            action_low=low,
            action_high=high,
        ).to("cpu")
        q1 = (
            q_class(3, 1, config["hidden_sizes"], config["activation"]).to("cpu")
            if q1_state_dict is not None
            else None
        )
        q2 = (
            q_class(3, 1, config["hidden_sizes"], config["activation"]).to("cpu")
            if q2_state_dict is not None
            else None
        )
    actor.load_state_dict(state_dict, strict=True)
    for model, state in ((q1, q1_state_dict), (q2, q2_state_dict)):
        if model is not None:
            model.load_state_dict(state, strict=True)
    for model in (actor, q1, q2):
        if model is not None:
            model.eval()
            model.requires_grad_(False)
    return InferenceModel(actor, q1, q2, source)


def load_agent(config: dict, env, checkpoint: str | Path, source_root=None):
    """Load trusted actor_state_dict; optional online Q heads, never optimizers."""
    path = Path(checkpoint)
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            "Checkpoint must be a regular, immutable trusted snapshot, not a symlink"
        )
    before = sha256(path)
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    after = sha256(path)
    if before != after:
        raise ValueError("Checkpoint changed while loading; use an immutable snapshot")
    if not isinstance(payload, dict) or "actor_state_dict" not in payload:
        raise ValueError("Expected SAC checkpoint containing actor_state_dict")
    embedded = payload.get("config")
    if embedded is None:
        embedded = payload.get("training_config")
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
            "normalize_rewards",
        ):
            defaults = {
                "env_kwargs": {},
                "reward_scale": 1.0,
                "reward_shaping": False,
                "normalize_rewards": False,
            }
            if key in embedded and embedded[key] != config.get(key, defaults.get(key)):
                raise ValueError(f"Checkpoint/config mismatch for {key}")
    model = build_agent(
        config,
        env,
        payload["actor_state_dict"],
        source_root,
        payload.get("q1_state_dict"),
        payload.get("q2_state_dict"),
    )
    metadata = {
        "path": str(path.resolve()),
        "sha256": before,
        "sha256_before_load": before,
        "sha256_after_load": after,
        "size_bytes": path.stat().st_size,
        "training_config": redact(embedded),
        "training_units": "embedded config, entropy-augmented training objective"
        if embedded
        else "unknown: legacy checkpoint has no embedded training config; caller YAML is not proof",
        "network_source": model.network_source,
        "q1_loaded": model.q1 is not None,
        "q2_loaded": model.q2 is not None,
        "soft_q_interpretation": SOFT_Q_NOTE,
    }
    for key in ("step", "global_step", "best_return"):
        value = payload.get(key)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = value.item()
        metadata[key] = (
            float(value)
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            else None
        )
    assert_checkpoint_unchanged(metadata)
    return model, metadata


def assert_checkpoint_unchanged(metadata: dict) -> None:
    if sha256(metadata["path"]) != metadata["sha256"]:
        raise ValueError(
            "Checkpoint changed during inference; use an immutable snapshot"
        )
    source = metadata.get("network_source")
    if source and sha256(source["path"]) != source["sha256"]:
        raise ValueError("Networks source changed during inference")


def deterministic_action(model: InferenceModel, obs) -> tuple[np.ndarray, dict]:
    """Native observations only; tanh(mean), never GaussianPolicy.sample()."""
    tensor = torch.as_tensor(np.asarray(obs, dtype=np.float32), device="cpu")
    if tensor.shape != (3,) or not torch.isfinite(tensor).all():
        raise ValueError("Expected three finite native observation features")
    tensor = tensor.unsqueeze(0)
    with torch.inference_mode():
        action_t = model.actor.deterministic(tensor)
        if action_t.shape != (1, 1) or not torch.isfinite(action_t).all():
            raise ValueError(
                "Unexpected or non-finite deterministic GaussianPolicy action"
            )
        values = {}
        for name, head in (("soft_q1", model.q1), ("soft_q2", model.q2)):
            if head is not None:
                prediction = head(tensor, action_t)
                if prediction.numel() != 1:
                    raise ValueError("Unexpected Q head output shape")
                values[name] = finite(prediction.item(), name)
            else:
                values[name] = None
    return action_t.cpu().numpy()[0].copy(), values


def native_source_metadata(env) -> dict:
    validate_env(env)
    base = env.unwrapped
    cls = type(base)
    try:
        source = inspect.getsource(cls.step)
        source_hash = hashlib.sha256(source.encode()).hexdigest()
    except (OSError, TypeError, AttributeError):
        source_hash = None
    physics = {}
    for name in ("g", "m", "l", "dt", "max_speed", "max_torque"):
        value = getattr(base, name, None)
        physics[name] = finite(value, name) if value is not None else None
    return {
        "class": f"{cls.__module__}.{cls.__name__}",
        "gymnasium_version": gym.__version__,
        "step_source_sha256": source_hash,
        "stock_engine_identified": cls.__module__
        == "gymnasium.envs.classic_control.pendulum",
        "observed_physics": physics,
        "native_horizon": NATIVE_HORIZON,
        "registered_limit": getattr(
            getattr(env, "spec", None), "max_episode_steps", None
        ),
        "evaluation_reward_scale": 1.0,
        "evaluation_reward_shaping": False,
        "reward_normalization": False,
        "timeout_note": "The normal Pendulum-v1 episode boundary is TimeLimit truncation after 200 transitions, not failure",
        "solved_threshold": None,
        "diagnostic_note": "Upright fraction and final-window settle snapshot are observer diagnostics, never actor inputs or termination rules",
    }


def pendulum_state(env) -> dict:
    """Observer only: read unwrapped.state, never recover hidden state from obs."""
    state = getattr(env.unwrapped, "state", None)
    if state is None:
        return {"theta": None, "theta_normalized": None, "angular_velocity": None}
    array = np.asarray(state)
    if array.shape != (2,):
        return {"theta": None, "theta_normalized": None, "angular_velocity": None}
    theta = finite(array[0], "native theta")
    velocity = finite(array[1], "native angular velocity")
    return {
        "theta": theta,
        "theta_normalized": (theta + math.pi) % (2 * math.pi) - math.pi,
        "angular_velocity": velocity,
    }


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    result = [0.0] * len(rewards)
    total = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        total = finite(rewards[index] + gamma * total, "discounted native return")
        result[index] = total
    return result


def upright_diagnostics(
    states: list[dict], dt: float | None, angle_limit=0.1, velocity_limit=0.5, window=20
) -> dict:
    """Post-transition occupancy and final-window snapshot, not mission success."""
    known = all(
        state["theta_normalized"] is not None and state["angular_velocity"] is not None
        for state in states
    )
    flags = (
        [
            abs(state["theta_normalized"]) <= angle_limit
            and abs(state["angular_velocity"]) <= velocity_limit
            for state in states
        ]
        if known and states
        else None
    )
    return {
        "known": flags is not None,
        "angle_limit_radians": angle_limit,
        "velocity_limit_radians_per_second": velocity_limit,
        "window_steps": window,
        "window_seconds": window * dt if dt is not None else None,
        "upright_flags": flags,
        "upright_fraction": sum(flags) / len(flags) if flags else None,
        "upright_time_seconds": sum(flags) * dt
        if flags is not None and dt is not None
        else None,
        "settle_snapshot": all(flags[-window:])
        if flags is not None and len(flags) >= window
        else None,
        "definition": "Post-transition |wrapped theta| <= 0.1 rad AND |velocity| <= 0.5 rad/s; settle_snapshot means all last 20 observations satisfy this, not lasting control or solved status",
    }


def rollout_episode(
    model: InferenceModel,
    env,
    config: dict,
    seed: int,
    episode: int = 0,
    max_steps: int | None = None,
    capture_frames: bool = False,
    source: dict | None = None,
) -> tuple[dict, list]:
    validate_counts(seed, 1, max_steps)
    validate_env(env)
    source = source if source is not None else native_source_metadata(env)
    limit = min(max_steps, NATIVE_HORIZON) if max_steps is not None else NATIVE_HORIZON
    obs, _ = env.reset(seed=seed)
    observations = [np.asarray(obs, dtype=np.float32).tolist()]
    states = [pendulum_state(env)]
    rewards, actions, q1_values, q2_values, step_metrics, frames = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    terminated = truncated = capped = False

    def capture():
        frame = env.render()
        if frame is None:
            raise RuntimeError(
                "rgb_array rendering returned no initial/transition/final frame"
            )
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB native rendering")
        frames.append(array.copy())

    if capture_frames:
        capture()
    while not (terminated or truncated or capped):
        action, values = deterministic_action(model, obs)
        torque = finite(action[0], "torque")
        low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
        excess = max(low - torque, torque - high, 0.0)
        applied = min(max(torque, low), high)
        metrics = {
            "t": len(rewards),
            "absolute_action": abs(torque),
            "action_lower_bound": low,
            "action_upper_bound": high,
            "over_bounds": excess > 0.0,
            "over_bounds_amount": excess,
            "native_action_cost": 0.001 * applied**2,
            "action_cost_definition": "0.001 * clipped native torque^2, separate diagnostic; no helper reward modification",
        }
        if excess > 0:
            raise ValueError(
                "Deterministic policy produced an out-of-bounds action; no helper clipping"
            )
        obs, reward, terminated, truncated, _ = env.step(action)
        terminated, truncated = bool(terminated), bool(truncated)
        rewards.append(finite(reward, "native reward"))
        actions.append(action.tolist())
        q1_values.append(values["soft_q1"])
        q2_values.append(values["soft_q2"])
        metrics.update(
            native_reward=rewards[-1], terminated=terminated, truncated=truncated
        )
        step_metrics.append(metrics)
        observation = np.asarray(obs, dtype=np.float32)
        if observation.shape != (3,) or not np.isfinite(observation).all():
            raise ValueError("Invalid next native observation")
        observations.append(observation.tolist())
        states.append(pendulum_state(env))
        capped = len(rewards) >= limit and not (terminated or truncated)
        if capture_frames:
            capture()
    targets = discounted_returns(rewards, float(config["gamma"]))
    diagnostic = upright_diagnostics(states[1:], source["observed_physics"].get("dt"))
    return {
        "episode": episode,
        "seed": seed,
        "native_return": finite(sum(rewards), "native return"),
        "episode_length": len(rewards),
        "native_discounted_return": targets[0],
        "native_rewards": rewards,
        "native_return_targets": targets,
        "target_definition": "Observed discounted native rewards to boundary, no entropy or bootstrap; TimeLimit/caller-cap targets are finite-horizon, not continuing soft-Q targets",
        "actions": actions,
        "observations": observations,
        "native_states": states,
        "theta_trajectory": [state["theta"] for state in states],
        "angular_velocity_trajectory": [state["angular_velocity"] for state in states],
        "soft_q1_predictions": q1_values,
        "soft_q2_predictions": q2_values,
        "soft_q_reward_points": [
            {
                "t": t,
                "soft_q1": q1_values[t],
                "soft_q2": q2_values[t],
                "native_discounted_reward_to_boundary": targets[t],
            }
            for t in range(len(rewards))
        ],
        "soft_q_interpretation": SOFT_Q_NOTE,
        "calibration_eligible": False,
        "step_metrics": step_metrics,
        "upright_diagnostic": diagnostic,
        "terminated": terminated,
        "truncated": truncated,
        "timeout": truncated and not terminated,
        "capped": capped,
        "complete_episode": not capped,
        "native_200_step_episode": len(rewards) == NATIVE_HORIZON
        and truncated
        and not terminated
        and not capped,
        "requested_max_steps": max_steps,
        "effective_cap": limit,
        "source_frames": len(frames),
        "final_observation": observations[-1],
    }, frames
