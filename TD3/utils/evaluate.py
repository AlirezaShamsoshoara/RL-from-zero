"""Evaluate every requested raw MountainCarContinuous episode, with offline W&B."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import wandb

from TD3.utils._common import (
    DEFAULT_CONFIG,
    ENV_ID,
    NATIVE_HORIZON,
    Q_NOTE,
    assert_checkpoint_unchanged,
    config_metadata,
    finite,
    load_agent,
    make_native_env,
    native_source_metadata,
    network_path,
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trusted immutable snapshot; omitted uses selected config inference_model_path",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Trusted stock root supplying exact TD3/td3/networks.py",
    )
    parser.add_argument("--episodes", type=positive_int, default=100)
    parser.add_argument("--seed", type=seed_int, default=180000)
    parser.add_argument("--out", type=Path, default=Path("TD3/logs/evaluation.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_counts(args.seed, args.episodes)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def summarize(episodes: list[dict]) -> dict:
    if not episodes:
        raise ValueError("Evaluation requires at least one episode")
    result = {
        "count": len(episodes),
        "mean_return_benchmark": 90.0,
        "benchmark_definition": "Mean native return >=90, not a per-episode physical goal predicate",
    }
    for key in ("native_return", "episode_length", "native_discounted_return"):
        values = np.asarray([finite(record[key], key) for record in episodes])
        result[key] = {
            "mean": finite(np.mean(values), key),
            "std": finite(np.std(values), key),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    for key in ("terminated", "truncated", "timeout", "capped", "complete_episode"):
        result[key + "_count"] = sum(bool(record[key]) for record in episodes)
    goals = [
        record["native_goal"]
        for record in episodes
        if record["native_goal"] is not None
    ]
    result.update(
        native_goal_known_count=len(goals),
        native_goal_count=sum(goals) if len(goals) == len(episodes) else None,
        native_goal_rate=sum(goals) / len(goals)
        if len(goals) == len(episodes)
        else None,
        failure_count=sum(not goal for goal in goals)
        if len(goals) == len(episodes)
        else None,
        mean_native_return_ge_90=result["native_return"]["mean"] >= 90.0,
    )
    for key in ("initial_q1", "initial_q2", "native_action_cost_sum"):
        known = [
            finite(record[key], key) for record in episodes if record[key] is not None
        ]
        result[key + "_known_count"] = len(known)
        result[key + "_mean"] = (
            finite(np.mean(known), key) if len(known) == len(episodes) else None
        )
    result["std_definition"] = "population standard deviation, ddof=0"
    return result


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
    # Complete individual records stay in local JSON, not huge nested W&B tables.
    for key in (
        "native_rewards",
        "native_return_targets",
        "position_trajectory",
        "velocity_trajectory",
        "q1_predictions",
        "q2_predictions",
        "native_action_costs",
    ):
        payload[f"eval/{key}"] = record[key]
    payload["eval/actions"] = [action[0] for action in record["actions"]]
    payload["eval/native_goal_known"] = record["native_goal"] is not None
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
        project=config.get("project", "rl-hero"),
        name=f"{config.get('run_name', 'td3-mountaincar')}-{name}",
        mode="offline",
        reinit=True,
        config={
            "algorithm": "TD3",
            "policy": "CPU Actor.forward, native observation only",
            "episodes": episodes,
            "seed": seed,
            "config_provenance": provenance,
            "checkpoint": checkpoint,
            "native_environment": source,
            "q_interpretation": Q_NOTE,
        },
    )


def evaluation_context(args):
    config = read_config(args.config)
    checkpoint_path = resolve_checkpoint(config, getattr(args, "checkpoint", None))
    source_root = getattr(args, "source_root", None)
    return config, checkpoint_path, source_root


def environment_metadata(env, config):
    source = native_source_metadata(env)
    source["training_env_kwargs"] = config.get("env_kwargs", {})
    source["ignored_training_max_episode_steps"] = config.get("env_kwargs", {}).get(
        "max_episode_steps"
    )
    source["ignored_training_max_steps_per_episode"] = config.get(
        "max_steps_per_episode"
    )
    source["requested_evaluation_cap"] = None
    return source


def result_metadata(config, provenance, checkpoint, source, run, records) -> dict:
    return {
        "schema_version": 1,
        "algorithm": "TD3",
        "env_id": ENV_ID,
        "device": "cpu",
        "policy": "Exact CPU Actor.forward, two native features only; no noise, pulses, extra info, scripts or hints",
        "config_provenance": provenance,
        "checkpoint": checkpoint,
        "native_environment": source,
        "native_horizon": NATIVE_HORIZON,
        "gamma": finite(config["gamma"], "gamma"),
        "q_interpretation": Q_NOTE,
        "max_steps": None,
        "return_units": "raw native undiscounted reward sum",
        "episodes": records,
        "seeds": [record["seed"] for record in records],
        "returns": [record["native_return"] for record in records],
        "episode_lengths": [record["episode_length"] for record in records],
        "wandb": {"mode": "offline", "run_id": run.id, "run_dir": run.dir},
    }


def evaluate(args: argparse.Namespace) -> dict:
    validate_counts(args.seed, args.episodes)
    config, checkpoint_path, source_root = evaluation_context(args)
    overwrite = getattr(args, "overwrite", False)
    protect_inputs(
        [args.out], [args.config, checkpoint_path, network_path(source_root)], overwrite
    )
    provenance = config_metadata(args.config, config)
    os.environ["WANDB_MODE"] = "offline"
    env, run = make_native_env(config), None
    try:
        model, checkpoint = load_agent(config, env, checkpoint_path, source_root)
        source = environment_metadata(env, config)
        run = initialize_offline_run(
            config,
            provenance,
            checkpoint,
            source,
            args.episodes,
            args.seed,
            "native-eval",
        )
        records = []
        for index in range(args.episodes):
            record, _ = rollout_episode(
                model, env, config, args.seed + index, episode=index, source=source
            )
            records.append(record)
            log_episode(run, record)
        summary = summarize(records)
        metrics = _scalar_metrics("eval/summary", summary)
        run.log(metrics, step=args.episodes)
        run.summary.update(metrics)
        assert_checkpoint_unchanged(checkpoint)
        result = result_metadata(config, provenance, checkpoint, source, run, records)
        result.update(
            selection="none: all requested episodes, including every timeout/failure",
            seed=args.seed,
            summary=summary,
        )
        write_json(args.out, result, overwrite)
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
    print(f"Wrote all {len(result['episodes'])} full native episodes to {args.out}")


if __name__ == "__main__":
    main()
