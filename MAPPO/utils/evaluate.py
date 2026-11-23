"""Evaluate all native MultiWalker episodes with JSON and offline W&B logging."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import wandb

from MAPPO.mappo.config import Config
from MAPPO.utils._common import (
    checkpoint_metadata,
    create_env,
    layout,
    load_agent,
    native_horizon,
    positive_int,
    provenance,
    redact,
    rollout_episode,
    safety_cap,
    seed_int,
    validate_seed_range,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("MAPPO/configs/multiwalker_delivery_long_credit.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Trusted checkpoint; defaults to cfg.inference_model_path.",
    )
    parser.add_argument("--episodes", type=positive_int, default=30)
    parser.add_argument("--seed", type=seed_int, default=10000)
    parser.add_argument("--out", type=Path, default=Path("MAPPO/logs/evaluation.json"))
    parser.add_argument(
        "--max-steps",
        type=positive_int,
        help="Optional safety cap, not an override of native max_cycles.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--random",
        action="store_true",
        help="Seeded random-action baseline; no checkpoint.",
    )
    group.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample the learned policy instead of deterministic actions.",
    )
    args = parser.parse_args(argv)
    try:
        validate_seed_range(args.seed, args.episodes)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def evaluate(args: argparse.Namespace) -> dict:
    cfg = Config.from_yaml(str(args.config))
    cfg.device = "cpu"
    os.environ["WANDB_MODE"] = "offline"
    env = create_env(cfg, args.seed)
    run = None
    try:
        ids, _, action_type = layout(env, cfg)
        validate_seed_range(args.seed, args.episodes, len(ids))
        horizon = native_horizon(env, cfg)
        cap = safety_cap(cfg, horizon, args.max_steps)
        agent, payload, metadata = None, None, None
        if not args.random:
            checkpoint = args.checkpoint or Path(cfg.inference_model_path)
            agent, payload = load_agent(cfg, env, checkpoint)
            metadata = checkpoint_metadata(checkpoint, payload)
        policy = (
            "random"
            if args.random
            else "sampled"
            if args.stochastic
            else "deterministic"
        )
        source = provenance(cfg, payload, args.seed, args.episodes, None)
        run = wandb.init(
            project=cfg.project,
            name=f"{cfg.run_name}-eval-{policy}",
            mode="offline",
            config=redact(
                {
                    "algorithm": "MAPPO",
                    "evaluation_policy": policy,
                    "action_type": action_type,
                    "provenance": source,
                    "native_max_cycles": horizon,
                    "safety_cap": cap,
                    "checkpoint": metadata,
                }
            ),
        )
        episodes = []
        for episode in range(args.episodes):
            record = rollout_episode(
                env,
                cfg,
                agent,
                args.seed + episode,
                cap,
                horizon,
                random_policy=args.random,
                deterministic=not args.stochastic,
            )
            record.pop("frames")
            episodes.append(record)
            print(
                f"Episode {episode + 1}/{args.episodes}: seed={record['seed']} "
                f"delivered={record.get('delivered')} reason={record.get('terminal_reason', 'unknown')} "
                f"steps={record['length']} remaining={record.get('remaining_distance')} "
                f"speed={record.get('package_speed_per_second')} team_return={record['team_return']:.2f}"
            )
            metrics = {
                "eval/episode": episode,
                "eval/seed": record["seed"],
                "eval/team_return": record["team_return"],
                "eval/discounted_team_return": record["discounted_team_return"],
                "eval/parallel_steps": record["length"],
                "eval/agent_transitions": record["agent_transitions"],
                "eval/terminated": int(record["terminated"]),
                "eval/truncated": int(record["truncated"]),
                "eval/capped": int(record["capped"]),
                "eval/survived_horizon": int(record["survived_horizon"]),
            }
            for key in (
                "initial_package_x",
                "final_package_x",
                "package_displacement",
                "goal_x",
                "remaining_distance",
                "progress_fraction",
                "delivered",
                "package_speed_per_second",
                "package_speed_per_parallel_step",
                "motion_fraction",
            ):
                value = record.get(key)
                if value is not None:
                    metrics[f"eval/{key}"] = value
            for name in ids:
                metrics[f"eval/{name}/return"] = record["per_agent_returns"][name]
                metrics[f"eval/{name}/discounted_return"] = record[
                    "per_agent_discounted_returns"
                ][name]
                value = record["initial_values"][name]
                if value is not None:
                    metrics[f"eval/{name}/initial_value"] = value
            wandb.log(metrics, step=episode)
        returns = [record["team_return"] for record in episodes]
        discounted = [record["discounted_team_return"] for record in episodes]
        displacements = [
            record["package_displacement"]
            for record in episodes
            if record.get("package_displacement") is not None
        ]
        known_delivery = [
            record.get("delivered")
            for record in episodes
            if record.get("delivered") is not None
        ]
        outcome_counts = {}
        for record in episodes:
            reason = record.get("terminal_reason", "unknown")
            outcome_counts[reason] = outcome_counts.get(reason, 0) + 1
        summary = {
            "delivery_known_count": len(known_delivery),
            "delivered_count": sum(value is True for value in known_delivery),
            "delivery_rate": float(np.mean(known_delivery))
            if len(known_delivery) == len(episodes)
            else None,
            "known_delivery_rate": float(np.mean(known_delivery))
            if known_delivery
            else None,
            "outcome_counts": outcome_counts,
            "count": len(returns),
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "min": float(np.min(returns)),
            "max": float(np.max(returns)),
            "discounted_return_mean": float(np.mean(discounted)),
            "discounted_return_std": float(np.std(discounted)),
            "mean_parallel_steps": float(
                np.mean([record["length"] for record in episodes])
            ),
            "survival_rate": float(
                np.mean([record["survived_horizon"] for record in episodes])
            ),
            "terminated_count": sum(record["terminated"] for record in episodes),
            "truncated_count": sum(record["truncated"] for record in episodes),
            "capped_count": sum(record["capped"] for record in episodes),
            "package_displacement_available_count": len(displacements),
            "package_displacement_mean": float(np.mean(displacements))
            if displacements
            else None,
            "package_displacement_min": float(np.min(displacements))
            if displacements
            else None,
            "package_displacement_max": float(np.max(displacements))
            if displacements
            else None,
        }
        for key in (
            "goal_x",
            "remaining_distance",
            "progress_fraction",
            "package_speed_per_second",
            "package_speed_per_parallel_step",
            "motion_fraction",
        ):
            values = [record[key] for record in episodes if record.get(key) is not None]
            summary[f"{key}_available_count"] = len(values)
            summary[f"{key}_mean"] = float(np.mean(values)) if values else None
            summary[f"{key}_min"] = float(np.min(values)) if values else None
            summary[f"{key}_max"] = float(np.max(values)) if values else None
        final_metrics = {
            f"eval/{key}": value for key, value in summary.items() if value is not None
        }
        wandb.log(final_metrics, step=args.episodes)
        run.summary.update(final_metrics)
        per_agent_summary = {}
        for name in ids:
            values = [record["per_agent_returns"][name] for record in episodes]
            per_agent_summary[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        result = {
            "algorithm": "MAPPO",
            "env_id": cfg.env_id,
            "policy": policy,
            "action_type": action_type,
            "device": "cpu",
            "seed": args.seed,
            "config_path": str(args.config.resolve()),
            "provenance": source,
            "agent_ids": ids,
            "native_max_cycles": horizon,
            "safety_cap": cap,
            "summary": summary,
            "per_agent_summary": per_agent_summary,
            "returns": returns,
            "discounted_returns": discounted,
            "lengths": [record["length"] for record in episodes],
            "seeds": [record["seed"] for record in episodes],
            "episodes": episodes,
            "team_return_definition": "sum over parallel steps of mean reward over original fixed agent IDs; missing rewards are zero",
            "discounted_return_definition": "sum_t gamma**t * native_team_reward_t, first reward at t=0",
            "length_units": "one simultaneous PettingZoo parallel step, not agent transitions",
            "delivery_definition": "No native failure, package x strictly beyond (terrain_length - 10) * (14/30), and all native get_last_dones true; not horizon survival",
            "delivery_rate_definition": "fraction of all requested episodes delivered when every outcome is known; otherwise null, with known_delivery_rate reported separately",
            "package_motion_definition": "native x samples every 20 parallel steps plus endpoints; average net displacement / simulation time (50 native frames per second)",
            "survival_definition": "no native failure and native max_cycles reached or truncation observed; not delivery",
            "success": None,
            "std_definition": "population standard deviation (ddof=0)",
            "critic_note": "Concatenated padded observations are used only for critic diagnostics; each deployed actor sees its own observation.",
            "checkpoint": metadata,
            "wandb": {"mode": "offline", "run_id": run.id, "run_dir": run.dir},
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return result
    finally:
        try:
            env.close()
        finally:
            if run is not None:
                run.finish()


def main(argv=None) -> None:
    args = parse_args(argv)
    print(json.dumps(evaluate(args), indent=2, allow_nan=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
