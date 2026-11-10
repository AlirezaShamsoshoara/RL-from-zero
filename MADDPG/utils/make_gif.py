"""Render trained MADDPG policies to an animated GIF (PettingZoo simple_spread).

Rolls out several evaluation episodes with the deterministic per-agent policies,
keeps the highest-returning ones, subsamples frames, and stitches them into a
compact GIF for the README demo. Must be run from the repository root so that
the ``MADDPG.*`` imports resolve (the ``make_gif.sh`` wrapper handles this).

Example:
    python -m MADDPG.utils.make_gif \
        --config MADDPG/configs/simple_spread.yaml \
        --checkpoint MADDPG/checkpoints/best.pt \
        --out MADDPG/assets/maddpg_simple_spread.gif
"""
from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import numpy as np

from MADDPG.maddpg.agent import MADDPGAgent
from MADDPG.maddpg.config import Config
from MADDPG.maddpg.utils import get_space_info, load_checkpoint, make_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="Path to the YAML config (defines env_id, network sizes, etc.).")
    p.add_argument("--checkpoint", required=True,
                   help="Path to the .pt checkpoint to load (use best.pt).")
    p.add_argument("--out", required=True, help="Output GIF path.")
    p.add_argument("--episodes", type=int, default=10,
                   help="Number of episodes to roll out before picking the best (default: 10).")
    p.add_argument("--keep-top", type=int, default=3,
                   help="How many of the best episodes to include in the GIF (default: 3).")
    p.add_argument("--fps", type=int, default=20,
                   help="Frames per second of the output GIF (default: 20).")
    p.add_argument("--stride", type=int, default=1,
                   help="Keep every Nth rendered frame to shrink the file (default: 1).")
    p.add_argument("--seed", type=int, default=2000,
                   help="Base seed; episode i uses seed+i (default: 2000).")
    return p.parse_args()


def _rollout(env, agent, obs_dims, seed: int):
    """Deterministic rollout of one PettingZoo parallel episode.

    Returns ``(episode_return, frames)`` where ``episode_return`` is the mean
    per-agent reward summed over the episode (matches the training metric)."""
    observations, _ = env.reset(seed=seed)
    agent_ids = list(observations.keys())
    frames = [env.render()]
    ret = 0.0
    max_steps = 200
    steps = 0
    while observations and steps < max_steps:
        obs_list = [
            observations.get(a, np.zeros(obs_dims[i], dtype=np.float32))
            for i, a in enumerate(agent_ids)
        ]
        actions_list = agent.act(obs_list, noise=0.0, deterministic=True)
        actions_dict = {a: actions_list[i] for i, a in enumerate(agent_ids)}
        observations, rewards, terminations, truncations, _ = env.step(actions_dict)
        frames.append(env.render())
        if rewards:
            ret += float(np.mean([rewards.get(a, 0.0) for a in agent_ids]))
        steps += 1
        if all(terminations.get(a, False) or truncations.get(a, False) for a in agent_ids):
            break
    return ret, frames


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)

    env = make_env(
        cfg.env_id, cfg.seed,
        n_agents=cfg.n_agents, max_cycles=cfg.max_cycles,
        render_mode="rgb_array",
    )
    n_agents, obs_dims, act_dims, action_lows, action_highs = get_space_info(env)

    agent = MADDPGAgent(
        n_agents=n_agents,
        obs_dims=obs_dims, act_dims=act_dims,
        action_lows=action_lows, action_highs=action_highs,
        hidden_sizes=cfg.hidden_sizes, activation=cfg.activation,
        actor_lr=cfg.actor_lr, critic_lr=cfg.critic_lr,
        gamma=cfg.gamma, tau=cfg.tau,
        target_policy_noise=cfg.target_policy_noise,
        target_noise_clip=cfg.target_noise_clip,
        device="cpu",
        grad_clip=cfg.grad_clip,
        critic_hidden_sizes=cfg.critic_hidden_sizes,
        layer_norm=cfg.layer_norm,
        twin_critics=cfg.twin_critics,
        policy_delay=cfg.policy_delay,
    )
    load_checkpoint(args.checkpoint, agent)
    for actor in agent.actors:
        actor.eval()

    episodes = []  # (return, frames)
    for ep in range(args.episodes):
        ret, frames = _rollout(env, agent, obs_dims, seed=args.seed + ep)
        episodes.append((ret, frames))
        print(f"ep {ep}: return={ret:.2f}  frames={len(frames)}")
    env.close()

    episodes.sort(key=lambda x: x[0], reverse=True)
    chosen = episodes[: args.keep_top]
    print("chosen returns:", [round(r, 2) for r, _ in chosen])

    out_frames = []
    for _, frames in chosen:
        out_frames.extend(frames[:: args.stride])
        out_frames.extend([frames[-1]] * 8)  # hold the final frame between episodes

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimsave(args.out, out_frames, fps=args.fps, loop=0)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved {args.out} ({size_mb:.2f} MB, {len(out_frames)} frames)")


if __name__ == "__main__":
    main()
