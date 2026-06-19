"""Render a trained IQL policy to an animated GIF.

Rolls out several evaluation episodes with the deterministic policy, keeps the
highest-returning ones, and stitches them into a compact GIF. Run from the
repository root so the ``IQL.*`` imports resolve (the ``make_gif.sh`` wrapper
handles this for you).

Example:
    python -m IQL.utils.make_gif \
        --config IQL/configs/pendulum_mixed.yaml \
        --checkpoint IQL/checkpoints_mixed/best.pt \
        --out IQL/assets/iql_pendulum.gif
"""
from __future__ import annotations

import argparse
import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from IQL.iql.agent import IQLAgent
from IQL.iql.config import Config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="Checkpoint to load. Use best.pt, not the last update.")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--keep-top", type=int, default=3)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--seed", type=int, default=4000)
    return p.parse_args()


def build_agent(cfg: Config, env: gym.Env) -> IQLAgent:
    return IQLAgent(
        obs_space=env.observation_space,
        act_space=env.action_space,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        value_lr=cfg.value_lr,
        gamma=cfg.gamma,
        expectile=cfg.expectile,
        temperature=cfg.temperature,
        max_weight=cfg.max_weight,
        tau=cfg.tau,
        device="cpu",
    )


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    env = gym.make(cfg.env_id, render_mode="rgb_array", **cfg.env_kwargs)
    agent = build_agent(cfg, env)
    agent.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    agent.actor.eval()

    episodes = []  # (return, frames)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ret = 0.0
        frames = []
        while not done:
            frames.append(env.render())
            action = agent.act(np.asarray(obs, dtype=np.float32), deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ret += float(reward)
        episodes.append((ret, frames))
        print(f"ep {ep}: return={ret:.1f} frames={len(frames)}")
    env.close()

    episodes.sort(key=lambda x: x[0], reverse=True)
    chosen = episodes[: args.keep_top]
    print("chosen returns:", [round(r, 1) for r, _ in chosen])

    out_frames = []
    for _, frames in chosen:
        out_frames.extend(frames[:: args.stride])
        out_frames.extend([frames[-1]] * 12)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimsave(args.out, out_frames, fps=args.fps, loop=0)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB, {len(out_frames)} frames)")


if __name__ == "__main__":
    main()
