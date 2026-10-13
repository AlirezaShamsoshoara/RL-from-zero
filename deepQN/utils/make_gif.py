"""Render deterministic, native-reward DQN episodes as a selected headless GIF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio

from deepQN.dqn.config import Config
from deepQN.dqn.utils import make_env, set_seed
from deepQN.utils._common import (
    episode_cap,
    load_agent,
    positive_int,
    rollout_episode,
    seed_int,
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
        help="Trusted best.pt checkpoint; defaults to cfg.inference_model_path.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("deepQN/assets/dqn_mountaincar.gif")
    )
    parser.add_argument("--episodes", type=positive_int, default=8)
    parser.add_argument("--keep-top", type=positive_int, default=3)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--stride", type=positive_int, default=2)
    parser.add_argument("--seed", type=seed_int, default=3000)
    args = parser.parse_args(argv)
    validate_seed_range(parser, args.seed, args.episodes)
    if args.keep_top > args.episodes:
        parser.error("keep-top must not exceed episodes")
    return args


def make_gif(args: argparse.Namespace) -> dict:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    cfg = Config.from_yaml(str(args.config))
    cfg.device = "cpu"
    cap = episode_cap(cfg)
    set_seed(args.seed)
    env = make_env(
        cfg.env_id, args.seed, render_mode="rgb_array", env_kwargs=cfg.env_kwargs
    )
    episodes = []
    try:
        checkpoint = args.checkpoint or Path(cfg.inference_model_path)
        agent, _ = load_agent(cfg, env, checkpoint)
        for episode in range(args.episodes):
            record = rollout_episode(
                env,
                agent,
                args.seed + episode,
                cap,
                cfg.gamma,
                cfg.env_id,
                capture_frames=True,
            )
            episodes.append(record)
            print(
                f"Episode {episode}: seed={record['seed']} native_return={record['return']:.2f} "
                f"steps={record['length']} success={record['success']} "
                f"truncated={record['truncated']} capped={record['capped']}"
            )
    finally:
        env.close()
    selected = sorted(episodes, key=lambda record: record["return"], reverse=True)[
        : args.keep_top
    ]
    all_returns = [record["return"] for record in episodes]
    selected_returns = [record["return"] for record in selected]
    print("All native returns:", all_returns)
    print("Selected native returns:", selected_returns)
    print(
        "Selected showcase, not unbiased evaluation; negative MountainCar return is not failure."
    )
    frames = []
    for record in selected:
        episode_frames = record["frames"]
        frames.extend(episode_frames[:: args.stride])
        if (len(episode_frames) - 1) % args.stride:
            frames.append(episode_frames[-1])
        frames.extend([episode_frames[-1]] * max(1, round(args.fps * 0.4)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(
        str(args.out), frames, format="pillow", duration=1000.0 / args.fps, loop=0
    )
    print(
        f"Saved {args.out} ({args.out.stat().st_size / 1e6:.2f} MB, {len(frames)} frames)"
    )
    return {
        "returns": all_returns,
        "selected_returns": selected_returns,
        "selected_seeds": [record["seed"] for record in selected],
        "selected_success": [record["success"] for record in selected],
        "frames": len(frames),
        "fps": args.fps,
    }


def main(argv=None) -> None:
    make_gif(parse_args(argv))


if __name__ == "__main__":
    main()
