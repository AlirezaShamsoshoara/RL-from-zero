"""Render deterministic PPO episodes to a headless, top-return demo GIF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from PPO.ppo.config import Config
from PPO.ppo.utils import set_seed
from PPO.utils._common import (
    deterministic_action,
    load_agent,
    positive_int,
    seed_int,
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
        help="Trusted PPO checkpoint; use best.pt, not the last snapshot.",
    )
    parser.add_argument("--out", type=Path, default=Path("PPO/assets/ppo_cartpole.gif"))
    parser.add_argument("--episodes", type=positive_int, default=8)
    parser.add_argument("--keep-top", type=positive_int, default=3)
    parser.add_argument("--seed", type=seed_int, default=3000)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--stride", type=positive_int, default=2)
    args = parser.parse_args(argv)
    validate_seed_range(parser, args.seed, args.episodes)
    if args.keep_top > args.episodes:
        parser.error("keep-top must not exceed episodes")
    return args


def make_gif(args: argparse.Namespace) -> dict:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    cfg = Config.from_yaml(str(args.config))
    set_seed(args.seed)
    env = gym.make(
        cfg.env_id, render_mode="rgb_array", **getattr(cfg, "env_kwargs", {})
    )
    episodes = []
    try:
        agent, _ = load_agent(cfg, env, args.checkpoint)
        for episode in range(args.episodes):
            seed = args.seed + episode
            env.action_space.seed(seed)
            obs, _ = env.reset(seed=seed)
            done = False
            episode_return = 0.0
            frames = []
            while not done:
                frame = env.render()
                if frame is None:
                    raise RuntimeError("rgb_array rendering returned no frame")
                frames.append(np.asarray(frame).copy())
                action = deterministic_action(agent, obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                episode_return += float(reward)
            if not np.isfinite(episode_return):
                raise ValueError(f"Non-finite return in episode {episode}")
            episodes.append((episode_return, seed, frames))
            print(
                f"Episode {episode}: seed={seed} return={episode_return:.2f} "
                f"frames={len(frames)}"
            )
    finally:
        env.close()

    all_returns = [ret for ret, _, _ in episodes]
    selected = sorted(episodes, key=lambda item: item[0], reverse=True)[: args.keep_top]
    selected_returns = [ret for ret, _, _ in selected]
    print("All returns:", all_returns)
    print("Selected returns:", selected_returns)
    frames = []
    for _, _, episode_frames in selected:
        frames.extend(episode_frames[:: args.stride])
        frames.extend([episode_frames[-1]] * max(1, round(args.fps * 0.4)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(args.out), frames, duration=1000.0 / args.fps, loop=0)
    print(
        f"Saved {args.out} ({args.out.stat().st_size / 1e6:.2f} MB, "
        f"{len(frames)} frames)"
    )
    return {
        "returns": all_returns,
        "selected_returns": selected_returns,
        "selected_seeds": [seed for _, seed, _ in selected],
        "frames": len(frames),
    }


def main(argv=None) -> None:
    make_gif(parse_args(argv))


if __name__ == "__main__":
    main()
