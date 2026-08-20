"""Render trained Nash Q-learning policies on grid soccer to a GIF.

Loads the joint-action Q-tables from a checkpoint, rolls out a chosen matchup
(vs random, vs the analytical Nash opponent, or self-play), and renders each
step via ``GridSoccerEnv.render``. Run from the repo root (the ``make_gif.sh``
wrapper puts the ``nash_ql`` package on the path).

Example:
    python Nash-QL/utils/make_gif.py \
        --config Nash-QL/configs/grid_soccer.yaml \
        --checkpoint Nash-QL/checkpoints_soccer/best.pt \
        --opponent random --keep win \
        --out Nash-QL/assets/nash_ql_grid_soccer.gif
"""
from __future__ import annotations

import argparse
import os
import random as _random
import sys

# Make the `nash_ql` package importable when run as a script.
_PKG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

import imageio.v2 as imageio
import numpy as np
import torch

from nash_ql.agent import NashQLearningAgent
from nash_ql.config import Config
from nash_ql.exact_solver import load_or_solve
from nash_ql.utils import make_env


_OPP_LABELS = {
    "random": "random",
    "exact": "analytical Nash",
    "self": "learned Nash (self-play)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True, help="Checkpoint with q_tables (use best.pt).")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=3, help="Number of games to concatenate into the GIF.")
    p.add_argument("--scan", type=int, default=60, help="Max seeds to scan for qualifying games.")
    p.add_argument("--opponent", choices=list(_OPP_LABELS), default="random",
                   help="Agent 1: 'random' shows the learned agent 0 attacking and "
                        "scoring; 'exact' plays the analytical Nash equilibrium (loads "
                        "the cached Shapley solution); 'self' is learned-vs-learned.")
    p.add_argument("--keep", choices=["any", "win", "loss", "draw"], default=None,
                   help="Which game outcomes (from agent 0's perspective) to keep. "
                        "Defaults: 'win' for random/exact/self.")
    p.add_argument("--dedupe-still", action="store_true",
                   help="Skip frames where the game state (positions + ball) is identical "
                        "to the previous frame. Useful for self-play, which quickly freezes "
                        "into a defensive standoff where nothing changes for many steps.")
    p.add_argument("--fps", type=int, default=3)
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

    agent = NashQLearningAgent(
        n_agents=env.n_agents, n_states=env.n_states, n_actions=env.n_actions,
        alpha=cfg.alpha, gamma=cfg.gamma,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_decay=1.0,
    )
    agent.Q = q_tables.copy()

    # For 'exact' we need the analytical Nash strategy for agent 1 at every state.
    # Loads from disk if cached (near-instant); otherwise runs Shapley VI (~3 min).
    exact_pi1 = None
    if args.opponent == "exact":
        soln = load_or_solve(
            env, cfg.env_kwargs, gamma=cfg.gamma,
            cache_dir=cfg.checkpoint_dir, shaping=0.0, tol=1e-6, verbose=True,
        )
        exact_pi1 = soln.pi1  # (n_states, n_actions), sampled at each step

    keep = args.keep or "win"
    opp_label = _OPP_LABELS[args.opponent]

    # Local RNG so mixed-strategy sampling (random opponent, analytical pi1*,
    # equilibrium sampling in greedy_actions) is reproducible per --seed.
    rng = _random.Random(args.seed)
    # `greedy_actions` samples via np.random.choice; seed that too when needed.
    np.random.seed(args.seed)

    frames = []
    kept = 0
    scanned = 0
    for seed_off in range(args.scan):
        scanned += 1
        states = env.reset(seed=args.seed + seed_off)
        title = f"Grid Soccer - A0 (Nash) vs A1 ({opp_label}) - game {kept + 1}"
        ep_frames = [env.render(title=f"{title}, step 0")]
        # Track the previous game state (positions + ball owner) for dedupe.
        prev_state_key = (tuple(env.positions[0]), tuple(env.positions[1]), env.ball_owner)
        steps = 0
        scorer = None
        while steps < cfg.max_steps_per_episode:
            if args.opponent == "random":
                a0 = agent.best_response_action(states[0], agent=0)
                a1 = rng.randint(0, env.n_actions - 1)
            elif args.opponent == "exact":
                a0 = agent.best_response_action(states[0], agent=0)
                a1 = int(rng.choices(range(env.n_actions), weights=exact_pi1[states[0]])[0])
            else:  # 'self': learned equilibrium on both sides
                a0, a1 = agent.greedy_actions(states)
            res = env.step([a0, a1])
            states = res.observations
            steps += 1
            state_key = (tuple(env.positions[0]), tuple(env.positions[1]), env.ball_owner)
            if not args.dedupe_still or state_key != prev_state_key:
                ep_frames.append(env.render(title=f"{title}, step {steps}"))
                prev_state_key = state_key
            if all(res.terminated):
                scorer = res.info.get("scorer")
                break
            if res.truncated:
                break

        outcome = "win" if scorer == 0 else ("loss" if scorer == 1 else "draw")
        if keep == "any" or keep == outcome:
            kept += 1
            frames.extend(ep_frames)
            frames.extend([ep_frames[-1]] * 4)  # hold the final frame between games
            if kept >= args.episodes:
                break
    print(f"kept {kept} games matching keep={keep} (scanned {scanned} seeds)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, loop=0)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
