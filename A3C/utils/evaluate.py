"""Evaluate all fresh-seed A3C episodes on native LunarLander, logging offline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import wandb

from A3C.utils._common import (
    config_metadata,
    finite,
    load_agent,
    make_native_env,
    native_source_metadata,
    positive_int,
    read_config,
    rollout_episode,
    seed_int,
    validate_counts,
    write_json,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("A3C/configs/lunarlander_greedy.yaml")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trusted, immutable snapshot; defaults to the selected config's inference_model_path",
    )
    parser.add_argument("--episodes", type=positive_int, default=30)
    parser.add_argument(
        "--seed",
        type=seed_int,
        default=180000,
        help="Fresh base seed; episode i resets with seed + i",
    )
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        default=None,
        help="Optional diagnostic cap, recorded separately from native TimeLimit",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional trusted stock repository root providing A3C/a3c/networks.py",
    )
    parser.add_argument("--out", type=Path, default=Path("A3C/logs/evaluation.json"))
    args = parser.parse_args(argv)
    try:
        validate_counts(args.seed, args.episodes, args.max_steps)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def summarize(episodes: list[dict]) -> dict:
    if not episodes:
        raise ValueError("Evaluation requires at least one episode")
    summary = {"count": len(episodes)}
    for field in ("native_return", "episode_length", "native_discounted_return"):
        values = np.asarray([finite(episode[field], field) for episode in episodes])
        summary[field] = {
            "mean": finite(np.mean(values), field),
            "std": finite(np.std(values), field),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    summary["benchmark_ge_200_fraction"] = float(
        np.mean([ep["benchmark_ge_200"] for ep in episodes])
    )
    for field in (
        "terminated",
        "truncated",
        "timeout",
        "capped",
        "calibration_eligible",
    ):
        summary[field + "_count"] = sum(bool(ep[field]) for ep in episodes)
    for field in (
        "native_landing",
        "landed_on_pad",
        "crash",
        "out_of_bounds",
        "failure",
    ):
        known = [ep[field] for ep in episodes if ep[field] is not None]
        summary[field + "_known_count"] = len(known)
        summary[field + "_count"] = sum(bool(value) for value in known)
        # Never report a known-subset fraction as the fraction over all episodes.
        summary[field + "_fraction"] = (
            float(np.mean(known)) if len(known) == len(episodes) else None
        )
    matched = [ep for ep in episodes if ep["calibration_eligible"]]
    summary["initial_value_vs_discounted_return"] = {
        "count": len(matched),
        "mean_error": finite(
            np.mean(
                [
                    ep["initial_native_value"] - ep["native_discounted_return"]
                    for ep in matched
                ]
            ),
            "value error",
        )
        if matched
        else None,
        "rmse": finite(
            np.sqrt(
                np.mean(
                    [
                        (ep["initial_native_value"] - ep["native_discounted_return"])
                        ** 2
                        for ep in matched
                    ]
                )
            ),
            "value RMSE",
        )
        if matched
        else None,
        "interpretation": "Matched initial-state native value versus observed argmax-policy discounted return. Training used a stochastic policy; this is a policy-shift diagnostic, not proof of value calibration.",
    }
    return summary


def _scalar_metrics(prefix: str, value: dict) -> dict:
    metrics = {}
    for key, item in value.items():
        name = f"{prefix}/{key}"
        if isinstance(item, dict):
            metrics.update(_scalar_metrics(name, item))
        elif isinstance(item, (int, float, bool)):
            metrics[name] = item
    return metrics


def evaluate(args: argparse.Namespace) -> dict:
    validate_counts(args.seed, args.episodes, getattr(args, "max_steps", None))
    config = read_config(args.config)
    if getattr(args, "checkpoint", None) is None:
        checkpoint_path = config.get("inference_model_path")
        if not checkpoint_path:
            raise ValueError(
                "Selected config must specify inference_model_path when --checkpoint is omitted"
            )
        args.checkpoint = Path(checkpoint_path)
    if Path(args.out).resolve() in {
        Path(args.config).resolve(),
        Path(args.checkpoint).resolve(),
    }:
        raise ValueError("Evaluation output must not overwrite config or checkpoint")
    provenance = config_metadata(args.config, config)
    os.environ["WANDB_MODE"] = "offline"
    env = make_native_env(config)
    run = None
    try:
        model, checkpoint = load_agent(
            config, env, args.checkpoint, getattr(args, "source_root", None)
        )
        source = native_source_metadata(env)
        source["training_env_kwargs"] = config.get("env_kwargs", {})
        source["ignored_training_max_episode_steps"] = config.get("env_kwargs", {}).get(
            "max_episode_steps"
        )
        source["effective_evaluation_cap"] = getattr(args, "max_steps", None)
        run = wandb.init(
            project=config.get("project", "rl-practice"),
            name=f"{config.get('run_name', 'a3c-lunarlander')}-native-eval",
            mode="offline",
            reinit=True,
            config={
                "algorithm": "A3C",
                "policy": "CPU categorical argmax",
                "episodes": args.episodes,
                "seed": args.seed,
                "max_steps": getattr(args, "max_steps", None),
                "training_config_provenance": provenance,
                "checkpoint": checkpoint,
                "native_environment": source,
            },
        )
        episodes = []
        for episode in range(args.episodes):
            record, _ = rollout_episode(
                model,
                env,
                config,
                args.seed + episode,
                episode=episode,
                max_steps=getattr(args, "max_steps", None),
                source=source,
            )
            episodes.append(record)
            # Every episode is logged, including crashes, truncations, and caps.
            metrics = _scalar_metrics("eval", record)
            metrics["eval/native_rewards"] = record["native_rewards"]
            metrics["eval/training_value_predictions"] = record[
                "training_value_predictions"
            ]
            metrics["eval/native_return_targets"] = record["native_return_targets"]
            run.log(metrics, step=episode)
        summary = summarize(episodes)
        metrics = _scalar_metrics("eval/summary", summary)
        run.log(metrics, step=args.episodes)
        run.summary.update(metrics)
        result = {
            "algorithm": "A3C",
            "env_id": config["env_id"],
            "device": "cpu",
            "policy": "deterministic categorical argmax; no policy RNG or info input",
            "config_provenance": provenance,
            "checkpoint": checkpoint,
            "source_root": str(Path(args.source_root).resolve())
            if getattr(args, "source_root", None)
            else None,
            "native_environment": source,
            "max_steps": getattr(args, "max_steps", None),
            "seed": args.seed,
            "seeds": [ep["seed"] for ep in episodes],
            "gamma": float(config["gamma"]),
            "return_units": "native undiscounted reward sum",
            "std_definition": "population standard deviation (ddof=0)",
            "benchmark_definition": "native episode return >= 200; not a physical landing predicate",
            "summary": summary,
            "episodes": episodes,
            "returns": [ep["native_return"] for ep in episodes],
            "episode_lengths": [ep["episode_length"] for ep in episodes],
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
    print(json.dumps(result["summary"], indent=2, allow_nan=False))
    print(f"Wrote all {len(result['episodes'])} episodes to {args.out}")


if __name__ == "__main__":
    main()
