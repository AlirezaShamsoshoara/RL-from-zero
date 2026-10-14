"""CPU-only, observation-only TD3 inference on raw native MountainCarContinuous."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path
import types

import gymnasium as gym
import numpy as np
import torch
import yaml

ENV_ID = "MountainCarContinuous-v0"
NATIVE_HORIZON = 999
Q_NOTE = (
    "TD3 Q heads were trained on noisy replay actions with a clipped noisy, "
    "smoothed target policy and bootstrap across TimeLimit. Observed deterministic "
    "Actor rewards stop at the native episode boundary. Policy and horizon differ; "
    "Q versus Monte Carlo points are not exact greedy finite-horizon calibration "
    "or proof of overestimation. Legacy checkpoint training semantics are unknown."
)
DEFAULT_CONFIG = Path("TD3/configs/mountaincar_balanced.yaml")
MAX_JSON_BYTES = 256 * 1024 * 1024
PHYSICS = {
    "min_action": -1.0,
    "max_action": 1.0,
    "min_position": -1.2,
    "max_position": 0.6,
    "max_speed": 0.07,
    "goal_position": 0.45,
    "goal_velocity": 0.0,
    "power": 0.0015,
}


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


def validate_counts(seed: int, episodes: int) -> None:
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if seed + episodes > 2**32:
        raise ValueError("seed + episodes - 1 must be less than 2**32")


def finite(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {name}")
    return number


def redact(value):
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
    raise ValueError("Provenance must contain only JSON-compatible values")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protect_inputs(outputs, inputs, overwrite=False) -> None:
    outputs = [Path(path) for path in outputs]
    inputs = [Path(path) for path in inputs if path is not None]
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Outputs must not overwrite each other")
    if set(resolved) & {path.resolve() for path in inputs}:
        raise ValueError("Outputs must not overwrite source files")
    for index, out in enumerate(outputs):
        if out.is_symlink():
            raise ValueError("Output symlinks are not allowed")
        if out.exists():
            if not out.is_file():
                raise ValueError("Outputs must be regular files")
            for other in outputs[index + 1 :] + inputs:
                if other.exists() and out.samefile(other):
                    raise ValueError(
                        "Output hard link aliases a source or another output"
                    )
            if not overwrite:
                raise ValueError(f"Output exists: {out}; use --overwrite deliberately")


def write_json(path: str | Path, data: dict, overwrite=False) -> None:
    # Compact serialization preserves full trajectories without 3D pretty dumps.
    text = json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"
    if len(text.encode("utf-8")) > MAX_JSON_BYTES:
        raise ValueError("JSON exceeds 256 MiB guard; use fewer episodes per output")
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Output symlinks are not allowed")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(text)


def native_env_kwargs(config: dict) -> dict:
    if config.get("env_id") != ENV_ID:
        raise ValueError(f"These helpers require {ENV_ID}")
    kwargs = config.get("env_kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("env_kwargs must be a mapping")
    unknown = set(kwargs) - {"goal_velocity", "max_episode_steps"}
    if unknown:
        raise ValueError("Unsupported native env_kwargs: " + ", ".join(sorted(unknown)))
    if finite(kwargs.get("goal_velocity", 0), "goal_velocity") != 0:
        raise ValueError("Native evaluation requires official default goal_velocity=0")
    limit = kwargs.get("max_episode_steps")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise ValueError("Training horizon override must be a positive integer")
    # Disclose a training TimeLimit override, but never apply it to raw evaluation.
    return {"goal_velocity": 0}


def read_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    native_env_kwargs(config)
    for key in ("gamma", "hidden_sizes", "activation"):
        if key not in config:
            raise ValueError(f"Selected config must explicitly specify {key}")
    if not 0 < finite(config["gamma"], "gamma") <= 1:
        raise ValueError("gamma must be in (0, 1]")
    hidden = config["hidden_sizes"]
    if not isinstance(hidden, (list, tuple)) or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in hidden
    ):
        raise ValueError("hidden_sizes must contain positive integers")
    if config["activation"] not in ("relu", "tanh", "elu", "leaky_relu"):
        raise ValueError("Unsupported activation")
    if finite(config.get("reward_scale", 1), "reward_scale") != 1:
        raise ValueError("Native evaluation requires unscaled rewards")
    for key in ("reward_shaping", "normalize_rewards", "normalize_observations"):
        if config.get(key, False) is not False:
            raise ValueError(f"Native evaluation rejects {key}")
    return config


def config_metadata(path: str | Path, config: dict) -> dict:
    clean = redact(config)
    return {
        "path": str(Path(path).resolve()),
        "config": clean,
        "redacted_sha256": hashlib.sha256(
            json.dumps(clean, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        "digest_note": "Hash covers redacted parsed config, not secret-bearing YAML bytes",
    }


def resolve_checkpoint(config: dict, checkpoint: str | Path | None) -> Path:
    if checkpoint is None:
        checkpoint = config.get("inference_model_path")
        if not checkpoint:
            raise ValueError("Selected config must specify inference_model_path")
    return Path(checkpoint)


def network_path(source_root=None) -> Path:
    root = (
        Path(source_root).resolve()
        if source_root
        else Path(__file__).resolve().parents[2]
    )
    return root / "TD3/td3/networks.py"


def validate_env(env) -> None:
    for name, width in (("observation_space", 2), ("action_space", 1)):
        space = getattr(env, name)
        if not isinstance(space, gym.spaces.Box) or space.shape != (width,):
            raise ValueError(f"Native {name} must be Box with shape ({width},)")
        if not np.isfinite(space.low).all() or not np.isfinite(space.high).all():
            raise ValueError(f"Non-finite native {name} bounds")
    if not np.array_equal(env.action_space.low, [-1.0]) or not np.array_equal(
        env.action_space.high, [1.0]
    ):
        raise ValueError("Native action bounds must be [-1, 1]")
    if getattr(getattr(env, "spec", None), "max_episode_steps", None) != NATIVE_HORIZON:
        raise ValueError(
            "Raw native evaluation requires the registered 999-step TimeLimit"
        )
    for key, expected in PHYSICS.items():
        value = getattr(env.unwrapped, key, None)
        if value is not None and finite(value, key) != expected:
            raise ValueError(f"Modified native physics: {key}")


def make_native_env(config: dict, render_mode=None):
    kwargs = native_env_kwargs(config)
    if gym.spec(ENV_ID).max_episode_steps != NATIVE_HORIZON:
        raise ValueError("Registered native horizon was modified; expected 999")
    env = gym.make(ENV_ID, render_mode=render_mode, **kwargs)
    try:
        validate_env(env)
    except Exception:
        env.close()
        raise
    return env


def _network_classes(source_root=None):
    path = network_path(source_root)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected regular trusted networks source: {path}")
    before = sha256(path)
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != before:
        raise ValueError("Networks source changed before import")
    module = types.ModuleType("_td3_cpu_networks_" + before[:16])
    module.__file__ = str(path)
    # Execute just trusted networks.py bytes, without writing stock __pycache__.
    with torch.random.fork_rng(devices=[]):
        exec(compile(source, str(path), "exec"), module.__dict__)
    if sha256(path) != before:
        raise ValueError("Networks source changed during import")
    return (
        module.Actor,
        module.Critic,
        {
            "path": str(path),
            "sha256": before,
            "explicit_stock_source": source_root is not None,
            "trust_note": "Exact trusted networks.py executes Python; not a sandbox",
        },
    )


@dataclass
class InferenceModel:
    actor: torch.nn.Module
    q1: torch.nn.Module | None = None
    q2: torch.nn.Module | None = None
    network_source: dict | None = None


def _validate_state(state, name):
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Expected nonempty {name} mapping")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must contain named tensors only")
        if (
            value.device.type != "cpu"
            or not value.is_floating_point()
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"{name} tensor must be finite floating CPU data: {key}")


def _load_strict(model, state):
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("Checkpoint model keys do not match exact networks source")
    for key, value in state.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"Checkpoint model shape/dtype mismatch: {key}")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    for value in model.state_dict().values():
        if value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError("Loaded model must remain finite on CPU")


def build_agent(
    config: dict,
    env,
    state_dict: dict,
    source_root=None,
    q1_state_dict=None,
    q2_state_dict=None,
) -> InferenceModel:
    """Construct Actor/Critic directly; no Agent, optimizer, targets, replay or seed."""
    validate_env(env)
    _validate_state(state_dict, "actor_state_dict")
    low = torch.as_tensor(env.action_space.low, dtype=torch.float32, device="cpu")
    high = torch.as_tensor(env.action_space.high, dtype=torch.float32, device="cpu")
    for key, expected in (
        ("action_scale", (high - low) / 2),
        ("action_bias", (high + low) / 2),
    ):
        actual = state_dict.get(key)
        if (
            actual is None
            or actual.shape != expected.shape
            or not torch.equal(actual, expected)
        ):
            raise ValueError(f"Checkpoint {key} disagrees with native action bounds")
    for name, state in (("q1", q1_state_dict), ("q2", q2_state_dict)):
        if state is not None:
            _validate_state(state, name)
    actor_class, critic_class, source = _network_classes(source_root)
    # fork_rng with no CUDA devices works with Torch >=2.0 and stock Actor code.
    with torch.random.fork_rng(devices=[]):
        actor = actor_class(
            2, 1, config["hidden_sizes"], config["activation"], low, high
        ).to("cpu")
        q1 = (
            critic_class(2, 1, config["hidden_sizes"], config["activation"]).to("cpu")
            if q1_state_dict is not None
            else None
        )
        q2 = (
            critic_class(2, 1, config["hidden_sizes"], config["activation"]).to("cpu")
            if q2_state_dict is not None
            else None
        )
    _load_strict(actor, state_dict)
    for model, state in ((q1, q1_state_dict), (q2, q2_state_dict)):
        if model is not None:
            _load_strict(model, state)
    return InferenceModel(actor, q1, q2, source)


def checkpoint_counters(payload: dict) -> dict:
    """Nested producer counters take precedence; never confuse critic and env steps."""
    nested = payload.get("metadata")
    nested = {} if nested is None else nested
    if not isinstance(nested, dict):
        raise ValueError("Checkpoint metadata must be a mapping")
    counters, sources = {}, {}
    candidates = {
        "environment_steps": (
            (nested, "environment_steps", "metadata"),
            (payload, "environment_steps", "top-level"),
            (payload, "step", "top-level"),
        ),
        "optimizer_updates": (
            (nested, "optimizer_updates", "metadata"),
            (nested, "critic_updates", "metadata"),
            (payload, "optimizer_updates", "top-level"),
            (payload, "critic_updates", "top-level"),
            (payload, "global_step", "legacy top-level"),
        ),
        "actor_updates": (
            (nested, "actor_updates", "metadata"),
            (payload, "actor_updates", "top-level"),
        ),
        "vector_steps": (
            (nested, "vector_steps", "metadata"),
            (payload, "vector_steps", "top-level"),
        ),
    }
    for name, options in candidates.items():
        counters[name], sources[name] = None, None
        for container, key, location in options:
            if key not in container:
                continue
            value = container[key]
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                value = value.item()
            if value is not None:
                number = finite(value, name)
                if number < 0 or int(number) != number:
                    raise ValueError(f"Checkpoint {name} must be a nonnegative integer")
                counters[name] = int(number)
            sources[name] = f"{location}.{key}"
            break
    counters["critic_updates"] = counters["optimizer_updates"]
    counters["counter_sources"] = sources
    counters["checkpoint_metadata"] = redact(nested)
    counters["counter_units"] = {
        "environment_steps": "Actual native transitions; top-level checkpoint step is a legacy alias",
        "vector_steps": "Vector step calls, not native transitions",
        "optimizer_updates": "Critic optimizer updates; legacy agent.global_step has these units",
        "critic_updates": "Alias of optimizer_updates, NOT environment_steps",
        "actor_updates": "Actual delayed actor updates if logged; never inferred from a ratio",
        "global_step": "Raw legacy agent critic-update counter, NOT checkpoint step or env transitions",
        "step": "Raw top-level checkpoint environment-transition count, NOT agent.global_step",
    }
    return counters


def load_agent(config: dict, env, checkpoint: str | Path, source_root=None):
    path = Path(checkpoint)
    if not path.is_file() or path.is_symlink():
        raise ValueError("Checkpoint must be a regular immutable trusted snapshot")
    before = sha256(path)
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    after = sha256(path)
    if before != after:
        raise ValueError("Checkpoint changed during load")
    if not isinstance(payload, dict) or "actor_state_dict" not in payload:
        raise ValueError("Expected actor_state_dict checkpoint payload")
    embedded = payload.get("config")
    if embedded is None:
        embedded = payload.get("training_config")
    if embedded is not None:
        if not isinstance(embedded, dict):
            raise ValueError("Checkpoint configuration provenance must be a mapping")
        defaults = {
            "env_kwargs": {},
            "reward_scale": 1,
            "reward_shaping": False,
            "normalize_rewards": False,
            "normalize_observations": False,
        }
        for key in (
            "env_id",
            "env_kwargs",
            "hidden_sizes",
            "activation",
            "gamma",
            "reward_scale",
            "reward_shaping",
            "normalize_rewards",
            "normalize_observations",
        ):
            if key in embedded and embedded[key] != config.get(key, defaults.get(key)):
                raise ValueError(f"Checkpoint/config mismatch: {key}")
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
        "training_provenance": "embedded fields checked; omitted fields unknown"
        if embedded
        else "unknown: legacy checkpoint has no embedded config; caller YAML is not historical evidence",
        "network_source": model.network_source,
        "q1_loaded": model.q1 is not None,
        "q2_loaded": model.q2 is not None,
        "q_interpretation": Q_NOTE,
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
    metadata.update(checkpoint_counters(payload))
    assert_checkpoint_unchanged(metadata)
    return model, metadata


def assert_checkpoint_unchanged(metadata: dict) -> None:
    if sha256(metadata["path"]) != metadata["sha256"]:
        raise ValueError("Checkpoint changed during inference")
    source = metadata.get("network_source")
    if source and sha256(source["path"]) != source["sha256"]:
        raise ValueError("Networks source changed during inference")


def _observation(obs) -> np.ndarray:
    raw = np.asarray(obs)
    if raw.shape != (2,) or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError("Expected two finite real native observation features")
    array = raw.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Native observation must remain finite in float32")
    return array


def deterministic_action(model: InferenceModel, obs) -> tuple[np.ndarray, dict]:
    tensor = torch.as_tensor(_observation(obs), device="cpu").unsqueeze(0)
    with torch.inference_mode():
        action = model.actor(tensor)
        if action.shape != (1, 1) or not torch.isfinite(action).all():
            raise ValueError("Invalid native Actor action shape/value")
        if not ((action >= -1) & (action <= 1)).all():
            raise ValueError("Actor action out of bounds; no helper clipping")
        values = {"q1": None, "q2": None}
        for name, head in (("q1", model.q1), ("q2", model.q2)):
            if head is not None:
                prediction = head(tensor, action)
                if prediction.shape != (1, 1):
                    raise ValueError("Invalid Critic output shape")
                values[name] = finite(prediction.item(), name)
    return action.cpu().numpy()[0].copy(), values


def native_source_metadata(env) -> dict:
    validate_env(env)
    base, cls = env.unwrapped, type(env.unwrapped)
    physics = {
        key: finite(getattr(base, key), key)
        if getattr(base, key, None) is not None
        else None
        for key in PHYSICS
    }
    source_hash, confirmed = None, False
    try:
        source = inspect.getsource(cls.step)
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        compact = "".join(source.split())
        confirmed = (
            cls.__module__ == "gymnasium.envs.classic_control.continuous_mountain_car"
            and cls.__name__ == "Continuous_MountainCarEnv"
            and all(value is not None for value in physics.values())
            and all(
                clause in compact
                for clause in (
                    "position>=self.goal_positionandvelocity>=self.goal_velocity",
                    "velocity+=force*self.power-0.0025*math.cos(3*position)",
                    "reward=100.0",
                    "reward-=math.pow(action[0],2)*0.1",
                    "returnself.state,reward,terminated,False,{}",
                )
            )
        )
    except (OSError, TypeError, AttributeError):
        pass
    return {
        "class": f"{cls.__module__}.{cls.__name__}",
        "gymnasium_version": gym.__version__,
        "step_source_sha256": source_hash,
        "goal_predicate_confirmed": confirmed,
        "observed_physics": physics,
        "native_horizon": NATIVE_HORIZON,
        "registered_limit": env.spec.max_episode_steps,
        "registered_reward_threshold": getattr(env.spec, "reward_threshold", None),
        "goal_definition": "native terminated AND engine position >= 0.45 AND velocity >= 0; no caller cap; never inferred from return",
        "reward_definition": "100 on native goal minus 0.1 * native action^2",
        "mean_return_benchmark": 90.0,
        "benchmark_note": "Native mean-return benchmark 90, distinct from each episode's physical goal predicate",
        "evaluation_reward_scale": 1.0,
        "evaluation_reward_shaping": False,
    }


def native_state(env, source: dict) -> dict:
    """Observer-only engine evidence; never inferred from return or actor observations."""
    unknown = {"position": None, "velocity": None}
    if not source.get("goal_predicate_confirmed", False):
        return unknown
    state = getattr(env.unwrapped, "state", None)
    if state is None:
        return unknown
    array = np.asarray(state)
    if array.shape != (2,):
        return unknown
    return {
        "position": finite(array[0], "engine position"),
        "velocity": finite(array[1], "engine velocity"),
    }


def native_outcome(
    env, terminated: bool, truncated: bool, capped: bool, source: dict
) -> dict:
    state = native_state(env, source)
    goal = None
    physics = source.get("observed_physics") or {}
    position_limit = physics.get("goal_position")
    velocity_limit = physics.get("goal_velocity")
    if (
        state["position"] is not None
        and state["velocity"] is not None
        and position_limit is not None
        and velocity_limit is not None
    ):
        goal = bool(
            terminated
            and not capped
            and state["position"] >= finite(position_limit, "goal_position")
            and state["velocity"] >= finite(velocity_limit, "goal_velocity")
        )
    return {
        "native_goal": goal,
        "failure": not goal if goal is not None else None,
        "timeout": bool(truncated and not terminated),
        "capped": bool(capped),
        "final_position": state["position"],
        "final_velocity": state["velocity"],
    }


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    gamma = finite(gamma, "gamma")
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    result, total = [0.0] * len(rewards), 0.0
    for index in range(len(rewards) - 1, -1, -1):
        total = finite(rewards[index] + gamma * total, "discounted native return")
        result[index] = total
    return result


def motion_diagnostics(states: list[dict]) -> dict:
    known = bool(states) and all(
        state["position"] is not None and state["velocity"] is not None
        for state in states
    )
    result = {
        "max_position": None,
        "max_absolute_velocity": None,
        "moving_fraction": None,
        "longest_motion_run_steps": None,
        "sustained_motion": None,
        "window_steps": 20,
        "velocity_threshold": 0.001,
        "definition": "Observer-only post-transition |velocity| >= 0.001 for 20 consecutive steps; not goal or policy input",
    }
    if known:
        flags = [abs(state["velocity"]) >= 0.001 for state in states]
        longest = current = 0
        for flag in flags:
            current = current + 1 if flag else 0
            longest = max(longest, current)
        result.update(
            max_position=max(state["position"] for state in states),
            max_absolute_velocity=max(abs(state["velocity"]) for state in states),
            moving_fraction=sum(flags) / len(flags),
            longest_motion_run_steps=longest,
            sustained_motion=longest >= 20,
        )
    return result


def rollout_episode(
    model: InferenceModel,
    env,
    config: dict,
    seed: int,
    episode: int = 0,
    capture_frames: bool = False,
    source: dict | None = None,
) -> tuple[dict, list]:
    """Full reset-to-native-goal/999 TimeLimit span; no extra cap or action script."""
    validate_counts(seed, 1)
    validate_env(env)
    source = source if source is not None else native_source_metadata(env)
    obs, _ = env.reset(seed=seed)
    observations = [_observation(obs).tolist()]
    states = [native_state(env, source)]
    rewards, actions, q1, q2, frames, costs = [], [], [], [], [], []
    terminated = truncated = False

    def capture():
        frame = np.asarray(env.render())
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Expected nonempty uint8 native RGB frame")
        if min(frame.shape[:2]) <= 0:
            raise ValueError("Empty native rendering")
        frames.append(frame.copy())

    if capture_frames:
        capture()
    while not (terminated or truncated):
        if len(rewards) >= NATIVE_HORIZON:
            raise ValueError(
                "Native engine failed to end at 999; no helper cap or fabricated completion"
            )
        action, values = deterministic_action(model, obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        terminated, truncated = bool(terminated), bool(truncated)
        rewards.append(finite(reward, "native reward"))
        actions.append([finite(action[0], "native action")])
        q1.append(values["q1"])
        q2.append(values["q2"])
        costs.append(
            0.1 * actions[-1][0] ** 2
            if source.get("goal_predicate_confirmed")
            else None
        )
        observations.append(_observation(obs).tolist())
        states.append(native_state(env, source))
        if capture_frames:
            capture()
    if truncated and not terminated and len(rewards) != NATIVE_HORIZON:
        raise ValueError(
            "Unexpected early TimeLimit; raw native evaluation requires 999"
        )
    targets = discounted_returns(rewards, config["gamma"])
    outcome = native_outcome(env, terminated, truncated, False, source)
    return {
        "episode": episode,
        "seed": seed,
        "native_return": finite(sum(rewards), "native episode return"),
        "episode_length": len(rewards),
        "native_discounted_return": targets[0],
        "native_rewards": rewards,
        "native_return_targets": targets,
        "observations": observations,
        "actions": actions,
        "native_states": states,
        "position_trajectory": [state["position"] for state in states],
        "velocity_trajectory": [state["velocity"] for state in states],
        "q1_predictions": q1,
        "q2_predictions": q2,
        "initial_q1": q1[0],
        "initial_q2": q2[0],
        "q_mc_points": [
            {
                "t": t,
                "q1": q1[t],
                "q2": q2[t],
                "native_discounted_reward_to_boundary": targets[t],
            }
            for t in range(len(rewards))
        ],
        "q_interpretation": Q_NOTE,
        "calibration_eligible": False,
        "target_definition": "Observed deterministic native discounted rewards to goal/TimeLimit; no bootstrap, no target noise",
        "native_action_costs": costs,
        "native_action_cost_sum": finite(sum(costs), "action cost")
        if all(cost is not None for cost in costs)
        else None,
        "action_bounds": [-1.0, 1.0],
        "over_bounds_count": 0,
        "absolute_action_mean": finite(
            np.mean(np.abs(actions)), "absolute action mean"
        ),
        "motion_diagnostics": motion_diagnostics(states[1:]),
        "terminated": terminated,
        "truncated": truncated,
        "complete_episode": True,
        "requested_max_steps": None,
        "native_horizon": NATIVE_HORIZON,
        "source_frames": len(frames),
        "final_observation": observations[-1],
        **outcome,
    }, frames
