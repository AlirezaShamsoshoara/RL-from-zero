"""Render trained Independent Q-learning policies on LineWorld to a GIF.

Loads the per-agent Q-tables from a checkpoint, rolls out the greedy joint
policy, and renders each step via ``LineWorldEnv.render``. Run from the repo root
(the ``make_gif.sh`` wrapper puts the ``independent_ql`` package on the path).

Example:
    python Independent-QL/utils/make_gif.py \
        --config Independent-QL/configs/line_world.yaml \
        --checkpoint Independent-QL/checkpoints/best.pt \
        --out Independent-QL/assets/independent_ql_lineworld.gif
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the `independent_ql` package importable when run as a script.
_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import imageio.v2 as imageio
import numpy as np
import torch

from independent_ql.agent import IndependentQLearningAgent
from independent_ql.config import Config
from independent_ql.utils import make_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="Checkpoint with q_tables (use best.pt).")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=3, help="Episodes to concatenate into the GIF.")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    env = make_env(cfg.env_id, cfg.seed, cfg.env_kwargs)

    data = torch.load(args.checkpoint, map_location="cpu")
    q_tables = data["q_tables"]
    if isinstance(q_tables, torch.Tensor):
        q_tables = q_tables.cpu().numpy()
    q_tables = np.asarray(q_tables, dtype=np.float32)

    agent = IndependentQLearningAgent(
        n_agents=env.n_agents, n_states=env.n_states, n_actions=env.n_actions,
        alpha=cfg.alpha, gamma=cfg.gamma,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0,
    )
    agent.Q = q_tables.copy()

    frames = []
    for ep in range(args.episodes):
        states = env.reset(seed=args.seed + ep)
        frames.append(env.render(title=f"LineWorld - episode {ep + 1}, step 0"))
        steps = 0
        while steps < cfg.max_steps_per_episode:
            actions = agent.greedy_actions(states)
            res = env.step(actions)
            states = res.observations
            steps += 1
            frames.append(env.render(title=f"LineWorld - episode {ep + 1}, step {steps}"))
            if all(res.terminated) or res.truncated:
                break
        frames.extend([frames[-1]] * 4)  # hold the final frame between episodes

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, loop=0)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
