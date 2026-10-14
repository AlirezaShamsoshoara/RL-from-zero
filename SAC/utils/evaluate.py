"""Evaluate every requested fresh-seed native SAC Pendulum episode offline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import wandb

from SAC.utils._common import (
    SOFT_Q_NOTE,
    assert_checkpoint_unchanged,
    config_metadata,
    finite,
    load_agent,
    make_native_env,
    native_source_metadata,
    positive_int,
    protect_inputs,
    read_config,
    resolve_checkpoint,
    rollout_episode,
    seed_int,
    validate_counts,
    write_json,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("SAC/configs/pendulum_fast.yaml")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trusted immutable snapshot; defaults to selected config inference_model_path",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional trusted stock root supplying SAC/sac/networks.py",
    )
    parser.add_argument("--episodes", type=positive_int, default=30)
    parser.add_argument(
        "--seed",
        type=seed_int,
        default=180000,
        help="Choose held-out seeds; episode i resets with seed + i",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Optional diagnostic cap, clipped to native 200; leave unset for full episodes",
    )
    parser.add_argument(
        "--diagnostic-return-threshold",
        type=float,
        default=None,
        help="Optional return benchmark, e.g. -200; NOT an official solved threshold",
    )
    parser.add_argument("--out", type=Path, default=Path("SAC/logs/evaluation.json"))
    args = parser.parse_args(argv)
    try:
        validate_counts(args.seed, args.episodes, args.max_steps)
        if args.diagnostic_return_threshold is not None:
            finite(args.diagnostic_return_threshold, "diagnostic threshold")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def summarize(episodes: list[dict], diagnostic_return_threshold=None) -> dict:
    if not episodes:
        raise ValueError("Evaluation requires at least one episode")
    summary = {
        "count": len(episodes),
        "solved_threshold": None,
        "timeout_interpretation": "Native 200-step TimeLimit is normal, not failure",
    }
    for field in ("native_return", "episode_length", "native_discounted_return"):
        values = np.asarray([finite(episode[field], field) for episode in episodes])
        summary[field] = {
            "mean": finite(np.mean(values), field),
            "std": finite(np.std(values), field),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    for field in (
        "terminated",
        "truncated",
        "timeout",
        "capped",
        "complete_episode",
        "native_200_step_episode",
    ):
        summary[field + "_count"] = sum(bool(episode[field]) for episode in episodes)
    for field in ("upright_fraction", "upright_time_seconds", "settle_snapshot"):
        known = [
            episode["upright_diagnostic"][field]
            for episode in episodes
            if episode["upright_diagnostic"][field] is not None
        ]
        summary[field + "_known_count"] = len(known)
        # Never relabel an available subset as an all-episode mean or success rate.
        summary[field + "_mean"] = (
            float(np.mean(known)) if len(known) == len(episodes) else None
        )
    metrics = [point for episode in episodes for point in episode["step_metrics"]]
    summary["action_diagnostics"] = {
        "absolute_action_mean": finite(
            np.mean([point["absolute_action"] for point in metrics]),
            "absolute action mean",
        ),
        "native_action_cost_sum": finite(
            sum(point["native_action_cost"] for point in metrics), "action cost sum"
        ),
        "over_bounds_count": sum(point["over_bounds"] for point in metrics),
        "validated_action_bounds": [-2.0, 2.0],
    }
    if diagnostic_return_threshold is not None:
        threshold = finite(diagnostic_return_threshold, "diagnostic threshold")
        summary["optional_return_diagnostic"] = {
            "threshold": threshold,
            "episode_count_ge_threshold": sum(
                episode["native_return"] >= threshold for episode in episodes
            ),
            "all_episode_mean_ge_threshold": summary["native_return"]["mean"]
            >= threshold,
            "interpretation": "User-requested numeric diagnostic only; Pendulum has no official solved threshold or mission-success count",
        }
    return summary


def _scalar_metrics(prefix: str, value: dict) -> dict:
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}"
        if isinstance(item, dict):
            result.update(_scalar_metrics(name, item))
        elif isinstance(item, (int, float, bool)):
            result[name] = item
    return result


def log_episode(run, record: dict) -> None:
    payload = _scalar_metrics("eval", record)
    for key in (
        "native_rewards",
        "native_return_targets",
        "theta_trajectory",
        "angular_velocity_trajectory",
        "actions",
        "soft_q1_predictions",
        "soft_q2_predictions",
    ):
        payload[f"eval/{key}"] = record[key]
    run.log(payload, step=record["episode"])


def initialize_offline_run(
    config: dict,
    provenance: dict,
    checkpoint: dict,
    source: dict,
    episodes: int,
    seed: int,
    name: str,
):
    os.environ["WANDB_MODE"] = "offline"
    return wandb.init(
        project=config.get("project", "rl-practice"),
        name=f"{config.get('run_name', 'sac-pendulum')}-{name}",
        mode="offline",
        reinit=True,
        config={
            "algorithm": "SAC",
            "policy": "CPU GaussianPolicy tanh(mean), observation only",
            "episodes": episodes,
            "seed": seed,
            "config_provenance": provenance,
            "checkpoint": checkpoint,
            "native_environment": source,
            "soft_q_interpretation": SOFT_Q_NOTE,
        },
    )


def evaluate(args: argparse.Namespace) -> dict:
    max_steps = getattr(args, "max_steps", None)
    validate_counts(args.seed, args.episodes, max_steps)
    config = read_config(args.config)
    checkpoint_path = resolve_checkpoint(config, getattr(args, "checkpoint", None))
    source_root = getattr(args, "source_root", None)
    networks_path = (
        Path(source_root) if source_root else Path(__file__).resolve().parents[2]
    ) / "SAC/sac/networks.py"
    protect_inputs([args.out], [args.config, checkpoint_path, networks_path])
    provenance = config_metadata(args.config, config)
    os.environ["WANDB_MODE"] = "offline"
    env = make_native_env(config)
    run = None
    try:
        model, checkpoint = load_agent(config, env, checkpoint_path, source_root)
        source = native_source_metadata(env)
        source["training_env_kwargs"] = config.get("env_kwargs", {})
        source["ignored_training_max_episode_steps"] = config.get("env_kwargs", {}).get(
            "max_episode_steps"
        )
        source["requested_evaluation_cap"] = max_steps
        source["effective_evaluation_cap"] = (
            min(max_steps, 200) if max_steps is not None else 200
        )
        run = initialize_offline_run(
            config,
            provenance,
            checkpoint,
            source,
            args.episodes,
            args.seed,
            "native-eval",
        )
        episodes = []
        for index in range(args.episodes):
            record, _ = rollout_episode(
                model,
                env,
                config,
                args.seed + index,
                episode=index,
                max_steps=max_steps,
                source=source,
            )
            episodes.append(record)
            log_episode(run, record)
        summary = summarize(
            episodes, getattr(args, "diagnostic_return_threshold", None)
        )
        metrics = _scalar_metrics("eval/summary", summary)
        run.log(metrics, step=args.episodes)
        run.summary.update(metrics)
        assert_checkpoint_unchanged(checkpoint)
        result = {
            "schema_version": 1,
            "algorithm": "SAC",
            "env_id": "Pendulum-v1",
            "device": "cpu",
            "policy": "GaussianPolicy.deterministic: tanh(mean) * action_scale + action_bias; no sampling or extra info input",
            "config_provenance": provenance,
            "checkpoint": checkpoint,
            "source_root": str(Path(source_root).resolve()) if source_root else None,
            "source_url": None,
            "source_id": None,
            "native_environment": source,
            "seed": args.seed,
            "seeds": [episode["seed"] for episode in episodes],
            "gamma": float(config["gamma"]),
            "max_steps": max_steps,
            "return_units": "native undiscounted reward sum",
            "std_definition": "population standard deviation (ddof=0)",
            "soft_q_interpretation": SOFT_Q_NOTE,
            "selection": "none: every requested episode is included",
            "summary": summary,
            "episodes": episodes,
            "returns": [episode["native_return"] for episode in episodes],
            "episode_lengths": [episode["episode_length"] for episode in episodes],
            "wandb": {"mode": "offline", "run_id": run.id, "run_dir": run.dir},
        }
        write_json(args.out, result)
        return result
    finally:
        try:
            env.close()
        finally:
            if run is not None:
                run.finish()


def main(argv=None) -> None:
    args = parse_args(argv)
    result = evaluate(args)
    print(
        json.dumps(
            {"all_returns": result["returns"], "summary": result["summary"]},
            indent=2,
            allow_nan=False,
        )
    )
    print(
        f"Wrote all {len(result['episodes'])} native episodes to {args.out}; no selection"
    )


if __name__ == "__main__":
    main()
