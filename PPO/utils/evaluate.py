"""Evaluate a PPO checkpoint or seeded random baseline without cloud logging."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import wandb

from PPO.ppo.config import Config
from PPO.ppo.utils import set_seed
from PPO.utils._common import (
    checkpoint_metadata,
    deterministic_action,
    load_agent,
    positive_int,
    seed_int,
    validate_env,
    validate_seed_range,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("PPO/configs/cartpole_tuned.yaml")
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("PPO/checkpoints_tuned/best.pt"),
        help="Trusted PPO checkpoint; use best.pt for the showcase.",
    )
    parser.add_argument("--episodes", type=positive_int, default=50)
    parser.add_argument(
        "--seed",
        type=seed_int,
        default=1000,
        help="Fresh base seed; episode i uses seed + i.",
    )
    parser.add_argument("--out", type=Path, default=Path("PPO/logs/evaluation.json"))
    parser.add_argument(
        "--random",
        action="store_true",
        help="Evaluate a seeded random policy; no checkpoint is loaded.",
    )
    args = parser.parse_args(argv)
    validate_seed_range(parser, args.seed, args.episodes)
    return args


def evaluate(args: argparse.Namespace) -> dict:
    cfg = Config.from_yaml(str(args.config))
    set_seed(args.seed)
    # Explicit mode overrides an inherited online or disabled W&B setting.
    os.environ["WANDB_MODE"] = "offline"
    env = gym.make(cfg.env_id, render_mode=None, **getattr(cfg, "env_kwargs", {}))
    run = None
    try:
        validate_env(env)
        agent = None
        metadata = None
        if not args.random:
            agent, payload = load_agent(cfg, env, args.checkpoint)
            metadata = checkpoint_metadata(args.checkpoint, payload)
        policy = "random" if args.random else "deterministic"
        run = wandb.init(
            project=cfg.project,
            name=f"{cfg.run_name}-eval-{policy}",
            mode="offline",
            config={
                "algorithm": "PPO",
                "evaluation_policy": policy,
                "evaluation_episodes": args.episodes,
                "evaluation_seed": args.seed,
                "training_config": cfg.to_dict(),
                "checkpoint": metadata,
            },
        )
        returns = []
        lengths = []
        seeds = []
        for episode in range(args.episodes):
            seed = args.seed + episode
            env.action_space.seed(seed)
            obs, _ = env.reset(seed=seed)
            done = False
            episode_return = 0.0
            length = 0
            while not done:
                action = (
                    env.action_space.sample()
                    if args.random
                    else deterministic_action(agent, obs)
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                length += 1
                done = bool(terminated or truncated)
            if not math.isfinite(episode_return):
                raise ValueError(f"Non-finite return in episode {episode}")
            returns.append(episode_return)
            lengths.append(length)
            seeds.append(seed)
            wandb.log(
                {
                    "eval/episode": episode,
                    "eval/seed": seed,
                    "eval/episode_return": episode_return,
                    "eval/episode_length": length,
                },
                step=episode,
            )
        summary = {
            "count": len(returns),
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "min": float(np.min(returns)),
            "max": float(np.max(returns)),
        }
        wandb.log(
            {f"eval/{key}": value for key, value in summary.items()}, step=args.episodes
        )
        run.summary.update({f"eval/{key}": value for key, value in summary.items()})
        result = {
            "algorithm": "PPO",
            "env_id": cfg.env_id,
            "policy": policy,
            "config_path": str(args.config.resolve()),
            "config": cfg.to_dict(),
            "device": "cpu",
            "seed": args.seed,
            "seeds": seeds,
            "return_units": "undiscounted episode reward sum",
            "std_definition": "population standard deviation (ddof=0)",
            "summary": summary,
            "returns": returns,
            "episode_lengths": lengths,
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
    result = evaluate(args)
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
