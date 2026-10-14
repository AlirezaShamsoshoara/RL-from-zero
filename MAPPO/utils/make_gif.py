"""Render deterministic continuous MultiWalker policies to a selected headless GIF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio

from MAPPO.mappo.config import Config
from MAPPO.utils._common import (
    create_env,
    layout,
    load_agent,
    native_horizon,
    positive_int,
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
        help="Trusted best.pt weights; defaults to cfg.inference_model_path.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("MAPPO/assets/mappo_multiwalker.gif")
    )
    parser.add_argument("--episodes", type=positive_int, default=8)
    parser.add_argument("--keep-top", type=positive_int, default=3)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--stride", type=positive_int, default=2)
    parser.add_argument("--seed", type=seed_int, default=30000)
    parser.add_argument("--max-steps", type=positive_int)
    parser.add_argument(
        "--require-delivery",
        action="store_true",
        help="Fail without writing a GIF if no episode truly delivers the native package.",
    )
    args = parser.parse_args(argv)
    if args.keep_top > args.episodes:
        parser.error("keep-top must not exceed episodes")
    try:
        validate_seed_range(args.seed, args.episodes)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def make_gif(args: argparse.Namespace) -> dict:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    cfg = Config.from_yaml(str(args.config))
    cfg.device = "cpu"
    env = create_env(cfg, args.seed, "rgb_array")
    episodes = []
    try:
        ids, _, action_type = layout(env, cfg)
        if action_type != "continuous":
            raise ValueError(
                "The MultiWalker GIF helper requires continuous actions (discretize_actions=false)"
            )
        validate_seed_range(args.seed, args.episodes, len(ids))
        horizon = native_horizon(env, cfg)
        cap = safety_cap(cfg, horizon, args.max_steps)
        checkpoint = args.checkpoint or Path(cfg.inference_model_path)
        agent, _ = load_agent(cfg, env, checkpoint)
        for episode in range(args.episodes):
            record = rollout_episode(
                env,
                cfg,
                agent,
                args.seed + episode,
                cap,
                horizon,
                deterministic=True,
                capture_frames=True,
            )
            episodes.append(record)
            print(
                f"Episode {episode}: seed={record['seed']} team_return={record['team_return']:.2f} "
                f"parallel_steps={record['length']} delivered={record.get('delivered')} "
                f"reason={record.get('terminal_reason', 'unknown')} "
                f"remaining={record.get('remaining_distance')} speed={record.get('package_speed_per_second')}"
            )
    finally:
        env.close()
    delivered_episodes = [
        record for record in episodes if record.get("delivered") is True
    ]
    if getattr(args, "require_delivery", False) and not delivered_episodes:
        raise ValueError(
            "No evaluated episode delivered the package beyond the unchanged native goal. "
            "No delivery GIF was written; inspect outcomes and match the config's "
            "native max_cycles to the intended full horizon."
        )
    eligible = delivered_episodes if delivered_episodes else episodes
    selected = sorted(eligible, key=lambda record: record["team_return"], reverse=True)[
        : args.keep_top
    ]
    returns = [record["team_return"] for record in episodes]
    selected_returns = [record["team_return"] for record in selected]
    print("All native team returns:", returns)
    print("Selected native team returns:", selected_returns)
    print("Selected showcase only; survival at the time limit is not delivery.")
    print(
        "Selected native delivery outcomes:",
        [record.get("delivered") for record in selected],
    )
    frames = []
    for record in selected:
        source = record["frames"]
        frames.extend(source[:: args.stride])
        if (len(source) - 1) % args.stride:
            frames.append(source[-1])
        frames.extend([source[-1]] * max(1, round(args.fps * 0.4)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        str(args.out), frames, format="pillow", duration=1000.0 / args.fps, loop=0
    )
    print(f"Saved {args.out} ({len(frames)} frames)")
    return {
        "returns": returns,
        "selected_returns": selected_returns,
        "selected_seeds": [record["seed"] for record in selected],
        "selected_survived_horizon": [
            record["survived_horizon"] for record in selected
        ],
        "selected_delivered": [record.get("delivered") for record in selected],
        "selected_terminal_reasons": [
            record.get("terminal_reason", "unknown") for record in selected
        ],
        "selected_goal_x": [record.get("goal_x") for record in selected],
        "frames": len(frames),
        "fps": args.fps,
    }


def main(argv=None) -> None:
    make_gif(parse_args(argv))


if __name__ == "__main__":
    main()
