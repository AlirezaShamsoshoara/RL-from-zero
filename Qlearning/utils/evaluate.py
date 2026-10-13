"""Fresh-seed tabular Q-learning evaluation with JSON and offline W&B logging."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import wandb

from Qlearning.ql.config import Config
from Qlearning.ql.utils import make_env, set_seed
from Qlearning.utils._common import (
    checkpoint_metadata,
    load_agent,
    positive_int,
    rollout_episode,
    seed_int,
    validate_config,
    validate_env,
    validate_seed_range,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("Qlearning/configs/frozenlake.yaml")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Trusted checkpoint; defaults to the config's inference_model_path.",
    )
    parser.add_argument("--episodes", type=positive_int, default=100)
    parser.add_argument("--seed", type=seed_int, default=10000)
    parser.add_argument(
        "--out", type=Path, default=Path("Qlearning/logs/evaluation.json")
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Seeded random-policy baseline; no checkpoint is loaded.",
    )
    args = parser.parse_args(argv)
    validate_seed_range(parser, args.seed, args.episodes)
    return args


def evaluate(args: argparse.Namespace) -> dict:
    cfg = Config.from_yaml(str(args.config))
    cfg.device = "cpu"
    validate_config(cfg)
    set_seed(args.seed)
    os.environ["WANDB_MODE"] = "offline"
    env = make_env(cfg.env_id, args.seed, render_mode=None, env_kwargs=cfg.env_kwargs)
    run = None
    try:
        validate_env(env)
        agent = None
        metadata = None
        if not args.random:
            checkpoint = args.checkpoint or Path(cfg.inference_model_path)
            agent, payload = load_agent(cfg, env, checkpoint)
            metadata = checkpoint_metadata(checkpoint, payload)
        policy = "random" if args.random else "deterministic_first_argmax"
        run = wandb.init(
            project=cfg.project,
            name=f"{cfg.run_name}-eval-{policy}",
            mode="offline",
            config={
                "algorithm": "Qlearning",
                "evaluation_policy": policy,
                "evaluation_episodes": args.episodes,
                "evaluation_seed": args.seed,
                "training_config": cfg.to_dict(),
                "checkpoint": metadata,
            },
        )
        episodes = []
        frozenlake = "FrozenLake" in cfg.env_id
        for episode in range(args.episodes):
            record = rollout_episode(
                env,
                agent,
                args.seed + episode,
                cfg.max_steps_per_episode,
                cfg.gamma,
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
                "eval/capped": int(record["capped"]),
            }
            if record["start_value"] is not None:
                metrics["eval/start_value"] = record["start_value"]
            if frozenlake:
                metrics["eval/success"] = int(record["return"] > 0.0)
            wandb.log(metrics, step=episode)
        returns = [record["return"] for record in episodes]
        discounted = [record["discounted_return"] for record in episodes]
        success_rate = float(np.mean(np.asarray(returns) > 0.0)) if frozenlake else None
        summary = {
            "count": len(returns),
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "min": float(np.min(returns)),
            "max": float(np.max(returns)),
            "discounted_return_mean": float(np.mean(discounted)),
            "discounted_return_std": float(np.std(discounted)),
            "mean_length": float(np.mean([record["length"] for record in episodes])),
            "capped_count": sum(record["capped"] for record in episodes),
            "success_rate": success_rate,
        }
        final_metrics = {
            f"eval/{key}": value for key, value in summary.items() if value is not None
        }
        wandb.log(final_metrics, step=args.episodes)
        run.summary.update(final_metrics)
        result = {
            "algorithm": "Qlearning",
            "env_id": cfg.env_id,
            "policy": policy,
            "config_path": str(args.config.resolve()),
            "config": cfg.to_dict(),
            "device": "cpu",
            "seed": args.seed,
            "max_steps_per_episode": cfg.max_steps_per_episode,
            "return_units": "undiscounted episode reward sum",
            "discounted_return_definition": "sum_t gamma**t * reward_t, t starts at zero",
            "start_value_definition": "max_a Q(initial_state, a); null for random baseline",
            "std_definition": "population standard deviation (ddof=0)",
            "success_definition": (
                "FrozenLake only: fraction of evaluated episodes with total return > 0"
                if frozenlake
                else None
            ),
            "interpretation": (
                "Fresh seeds do not change a fixed deterministic FrozenLake map or "
                "start state. Same-map repetitions are not stochastic or map generalization."
            ),
            "summary": summary,
            "returns": returns,
            "discounted_returns": discounted,
            "start_values": [record["start_value"] for record in episodes],
            "episode_lengths": [record["length"] for record in episodes],
            "seeds": [record["seed"] for record in episodes],
            "episodes": episodes,
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
