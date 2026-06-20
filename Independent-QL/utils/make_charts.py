"""Generate Independent-QL training charts (chart_01/02/03.png) from a wandb run.

Reads the metric history straight from a run's ``.wandb`` datastore file (works
for offline runs too, no network needed) and writes three PNGs:

  - chart_01: mean episode return across agents (the learning curve)
  - chart_02: per-agent mean returns
  - chart_03: epsilon (exploration) schedule

Example:
    python Independent-QL/utils/make_charts.py \
        --run wandb/offline-run-XXXX/run-XXXX.wandb \
        --out-dir Independent-QL/assets --title "Independent-QL LineWorld"
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Path to run-<id>.wandb")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--smooth", type=int, default=11)
    return p.parse_args()


def load_history(run_path: str) -> list:
    ds = datastore.DataStore()
    ds.open_for_scan(run_path)
    rows = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") == "history":
            row = {}
            for it in rec.history.item:
                nk = list(it.nested_key)
                key = "/".join(nk) if nk else it.key
                try:
                    row[key] = json.loads(it.value_json)
                except Exception:
                    row[key] = it.value_json
            rows.append(row)
    rows = [r for r in rows if "progress/episode" in r]
    rows.sort(key=lambda r: r["progress/episode"])
    return rows


def smooth(y: np.ndarray, w: int) -> np.ndarray:
    y = np.asarray(y, float)
    out = np.copy(y)
    for i in range(len(y)):
        lo = max(0, i - w // 2)
        hi = min(len(y), i + w // 2 + 1)
        seg = y[lo:hi]
        seg = seg[~np.isnan(seg)]
        out[i] = np.mean(seg) if len(seg) else np.nan
    return out


def main() -> None:
    args = parse_args()
    rows = load_history(args.run)
    if not rows:
        raise SystemExit(f"No history rows found in {args.run}")
    os.makedirs(args.out_dir, exist_ok=True)
    suffix = args.title

    ep = np.array([r["progress/episode"] for r in rows], float)
    mean_ret = np.array([r.get("charts/mean_return", np.nan) for r in rows], float)
    eps = np.array([r.get("charts/epsilon", np.nan) for r in rows], float)
    n_agents = sum(1 for k in rows[-1] if k.startswith("charts/agent") and k.endswith("_mean_return"))
    agent_ret = {
        i: np.array([r.get(f"charts/agent{i}_mean_return", np.nan) for r in rows], float)
        for i in range(max(n_agents, 1))
    }

    plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.unicode_minus": False})

    # chart_01 - mean return across agents
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ep, mean_ret, color=PALETTE[2], lw=0.8, alpha=0.35, label="Mean return (raw)")
    ax.plot(ep, smooth(mean_ret, args.smooth), color=PALETTE[2], lw=2.0, label="Mean return (smoothed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean return across agents")
    ax.set_title(f"Team learning curve - {suffix}")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_01.png"); plt.close(fig)

    # chart_02 - per-agent returns
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, y in agent_ret.items():
        ax.plot(ep, smooth(y, args.smooth), color=PALETTE[i % len(PALETTE)], lw=1.8, label=f"agent {i}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean return (per agent)")
    ax.set_title(f"Per-agent returns - {suffix}")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_02.png"); plt.close(fig)

    # chart_03 - epsilon schedule
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ep, eps, color=PALETTE[4], lw=1.8, label="Epsilon (exploration)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title(f"Exploration schedule - {suffix}")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_03.png"); plt.close(fig)

    print(f"Charts written to {args.out_dir} | points={len(rows)} | "
          f"final_mean_return={mean_ret[-1]:.2f} peak={np.nanmax(mean_ret):.2f}")


if __name__ == "__main__":
    main()
