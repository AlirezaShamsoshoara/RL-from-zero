"""Evaluate native-reward DQN policies with JSON and forced offline W&B logs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import wandb

from deepQN.dqn.config import Config
from deepQN.dqn.utils import make_env, set_seed
from deepQN.utils._common import (
    checkpoint_metadata,
    episode_cap,
    load_agent,
    positive_int,
    redact_config,
    rollout_episode,
    seed_int,
    validate_env,
    validate_seed_range,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("deepQN/configs/mountaincar_exploration.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Trusted checkpoint; defaults to cfg.inference_model_path.",
    )
    parser.add_argument("--episodes", type=positive_int, default=100)
    parser.add_argument("--seed", type=seed_int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("deepQN/logs/evaluation.json"))
    parser.add_argument(
        "--random",
        action="store_true",
        help="Seeded random-action baseline; no checkpoint is loaded.",
    )
    args = parser.parse_args(argv)
    validate_seed_range(parser, args.seed, args.episodes)
    return args


def evaluate(args: argparse.Namespace) -> dict:
    cfg = Config.from_yaml(str(args.config))
    cfg.device = "cpu"
    cap = episode_cap(cfg)
    set_seed(args.seed)
    os.environ["WANDB_MODE"] = "offline"
    env = make_env(cfg.env_id, args.seed, render_mode=None, env_kwargs=cfg.env_kwargs)
    run = None
    try:
        validate_env(env)
        agent = None
        metadata = None
        evaluation_config = redact_config(cfg.to_dict())
        checkpoint_training_config = None
        config_mismatches = {}
        if not args.random:
            checkpoint = args.checkpoint or Path(cfg.inference_model_path)
            agent, payload = load_agent(cfg, env, checkpoint)
            metadata = checkpoint_metadata(checkpoint, payload)
            saved_config = payload.get("config")
            if saved_config is not None and not isinstance(saved_config, dict):
                raise ValueError("checkpoint training config must be a mapping")
            checkpoint_training_config = redact_config(saved_config)
            if checkpoint_training_config is not None:
                for key in (
                    "env_id",
                    "env_kwargs",
                    "hidden_sizes",
                    "activation",
                    "gamma",
                    "double_dqn",
                    "normalize_observations",
                    "total_steps",
                    "lr",
                    "learning_starts",
                    "train_freq",
                    "target_update_interval",
                    "epsilon_start",
                    "epsilon_end",
                    "epsilon_decay_steps",
                    "exploration_hold_min",
                    "exploration_hold_max",
                ):
                    if key in checkpoint_training_config and checkpoint_training_config[
                        key
                    ] != evaluation_config.get(key):
                        config_mismatches[key] = {
                            "training": checkpoint_training_config[key],
                            "evaluation": evaluation_config.get(key),
                        }
        policy = "random" if args.random else "deterministic_argmax"
        run = wandb.init(
            project=cfg.project,
            name=f"{cfg.run_name}-eval-{policy}",
            mode="offline",
            config={
                "algorithm": "DQN",
                "evaluation_policy": policy,
                "evaluation_episodes": args.episodes,
                "evaluation_seed": args.seed,
                "evaluation_config": evaluation_config,
                "checkpoint_training_config": checkpoint_training_config,
                "config_mismatches": config_mismatches,
                "checkpoint": metadata,
            },
        )
        episodes = []
        for episode in range(args.episodes):
            record = rollout_episode(
                env,
                agent,
                args.seed + episode,
                cap,
                cfg.gamma,
                cfg.env_id,
                random_policy=args.random,
            )
            record.pop("frames")
            episodes.append(record)
            metrics = {
                "eval/episode": episode,
                "eval/seed": record["seed"],
                "eval/episode_return": record["return"],
                "eval/discounted_return": record["discounted_return"],
                "eval/episode_length": record["length"],
                "eval/terminated": int(record["terminated"]),
                "eval/truncated": int(record["truncated"]),
                "eval/capped": int(record["capped"]),
            }
            for key in ("initial_q", "max_position", "success"):
                if record[key] is not None:
                    metrics[f"eval/{key}"] = record[key]
            wandb.log(metrics, step=episode)
        returns = [record["return"] for record in episodes]
        discounted = [record["discounted_return"] for record in episodes]
        mountaincar = cfg.env_id == "MountainCar-v0"
        summary = {
            "count": len(returns),
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "min": float(np.min(returns)),
            "max": float(np.max(returns)),
            "discounted_return_mean": float(np.mean(discounted)),
            "discounted_return_std": float(np.std(discounted)),
            "mean_length": float(np.mean([record["length"] for record in episodes])),
            "termination_rate": float(
                np.mean([record["terminated"] for record in episodes])
            ),
            "truncation_rate": float(
                np.mean([record["truncated"] for record in episodes])
            ),
            "capped_count": sum(record["capped"] for record in episodes),
            "success_rate": (
                float(np.mean([record["success"] for record in episodes]))
                if mountaincar
                else None
            ),
            "max_position": (
                max(record["max_position"] for record in episodes)
                if mountaincar
                else None
            ),
        }
        final_metrics = {
            f"eval/{key}": value for key, value in summary.items() if value is not None
        }
        wandb.log(final_metrics, step=args.episodes)
        run.summary.update(final_metrics)
        spec = getattr(env, "spec", None)
        result = {
            "algorithm": "DQN",
            "env_id": cfg.env_id,
            "policy": policy,
            "config_path": str(args.config.resolve()),
            "config": evaluation_config,
            "evaluation_config": evaluation_config,
            "checkpoint_training_config": checkpoint_training_config,
            "config_mismatches": config_mismatches,
            "device": "cpu",
            "seed": args.seed,
            "max_steps_per_episode": cap,
            "native_max_episode_steps": getattr(spec, "max_episode_steps", None),
            "return_units": "native undiscounted episode reward sum, no shaping",
            "discounted_return_definition": "sum_t gamma**t * native_reward_t, first reward at t=0",
            "initial_q_definition": "max_a online_Q(initial_observation, a); null for random baseline",
            "std_definition": "population standard deviation (ddof=0)",
            "success_definition": (
                "MountainCar-v0 only: true termination means reaching the goal; "
                "negative native return is not a failure criterion"
                if mountaincar
                else None
            ),
            "summary": summary,
            "episodes": episodes,
            "returns": returns,
            "discounted_returns": discounted,
            "episode_lengths": [record["length"] for record in episodes],
            "seeds": [record["seed"] for record in episodes],
            "initial_q_values": [record["initial_q"] for record in episodes],
            "max_positions": [record["max_position"] for record in episodes],
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
