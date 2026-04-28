"""Render a trained DDPG policy to an animated GIF.

Runs several evaluation episodes with the deterministic policy, keeps the
highest-returning ones, and stitches them into a compact GIF — handy for the
README demo. Must be run from the repository root so that the ``DDPG.*``
imports resolve (the ``make_gif.sh`` wrapper handles this for you).

Example:
    python -m DDPG.utils.make_gif \
        --config DDPG/configs/lunarlander_continuous_tuned.yaml \
        --checkpoint DDPG/checkpoints_tuned/best.pt \
        --out DDPG/assets/ddpg_lunarlander.gif
"""
from __future__ import annotations

import argparse
import os

import gymnasium as gym
import imageio.v2 as imageio
import torch

from DDPG.ddpg.agent import DDPGAgent
from DDPG.ddpg.config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="Path to the YAML config (defines env_id, network sizes, etc.).")
    p.add_argument("--checkpoint", required=True,
                   help="Path to the .pt checkpoint to load (use best.pt, not the last one).")
    p.add_argument("--out", required=True,
                   help="Output GIF path.")
    p.add_argument("--episodes", type=int, default=8,
                   help="Number of episodes to roll out before picking the best (default: 8).")
    p.add_argument("--keep-top", type=int, default=3,
                   help="How many of the best episodes to include in the GIF (default: 3).")
    p.add_argument("--fps", type=int, default=30,
                   help="Frames per second of the output GIF (default: 30).")
    p.add_argument("--stride", type=int, default=2,
                   help="Keep every Nth frame to shrink the file (default: 2).")
    p.add_argument("--seed", type=int, default=2000,
                   help="Base seed; episode i uses seed+i (default: 2000).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    env = gym.make(cfg.env_id, render_mode="rgb_array", **cfg.env_kwargs)
    agent = DDPGAgent(
        env.observation_space, env.action_space, cfg.hidden_sizes, cfg.activation,
        cfg.actor_lr, cfg.critic_lr, cfg.gamma, cfg.tau,
        cfg.target_policy_noise, cfg.target_noise_clip, "cpu",
    )
    agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    episodes = []  # list of (return, frames)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ret = 0.0
        frames = []
        while not done:
            frames.append(env.render())
            action = agent.act(obs, noise=0.0, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ret += reward
        episodes.append((ret, frames))
        print(f"ep {ep}: return={ret:.1f} frames={len(frames)}")
    env.close()

    episodes.sort(key=lambda x: x[0], reverse=True)
    chosen = episodes[: args.keep_top]
    print("chosen returns:", [round(r, 1) for r, _ in chosen])

    out_frames = []
    for _, frames in chosen:
        out_frames.extend(frames[:: args.stride])
        out_frames.extend([frames[-1]] * 12)  # hold the final frame (~0.4s) between episodes

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimsave(args.out, out_frames, fps=args.fps, loop=0)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved {args.out} ({size_mb:.2f} MB, {len(out_frames)} frames)")


if __name__ == "__main__":
    main()
