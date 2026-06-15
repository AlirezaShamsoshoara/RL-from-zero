"""Generate DDPG training charts (chart_01/02/03.png) from a local wandb run.

Reads the metric history straight from a run's ``.wandb`` datastore file (works
for offline runs too — no network needed) and writes three PNGs:

  - chart_01: critic Q-value vs. realized return (the over-estimation story)
  - chart_02: actor & critic loss
  - chart_03: episode return & length

Example:
    python -m DDPG.utils.make_charts \
        --run wandb/offline-run-XXXX/run-XXXX.wandb \
        --out-dir DDPG/assets \
        --title "DDPG + target smoothing"
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

BLUE, RED, GREEN, ORANGE = "#2563eb", "#dc2626", "#16a34a", "#ea580c"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                   help="Path to a run's .wandb file (e.g. wandb/<run>/run-<id>.wandb).")
    p.add_argument("--out-dir", required=True,
                   help="Directory to write chart_01/02/03.png into.")
    p.add_argument("--title", default="",
                   help="Suffix appended to each chart title (e.g. the run name).")
    p.add_argument("--smooth", type=int, default=9,
                   help="Moving-average window (in logging points) for smoothed lines (default: 9).")
    return p.parse_args()


def load_history(run_path: str) -> list[dict]:
    """Scan the wandb datastore and return logged history rows as dicts."""
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
                # wandb stores keys either flat (`key`) or nested (`nested_key`,
                # a repeated field) — join the latter with "/".
                nk = list(it.nested_key)
                key = "/".join(nk) if nk else it.key
                try:
                    row[key] = json.loads(it.value_json)
                except Exception:
                    row[key] = it.value_json
            rows.append(row)
    rows = [r for r in rows if "progress/step" in r]
    rows.sort(key=lambda r: r["progress/step"])
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

    steps = np.array([r["progress/step"] for r in rows])
    ret = np.array([r.get("charts/avg_return", np.nan) for r in rows], float)
    length = np.array([r.get("charts/avg_length", np.nan) for r in rows], float)
    q = np.array([r.get("stats/q_value", np.nan) for r in rows], float)
    closs = np.array([r.get("loss/critic_loss", np.nan) for r in rows], float)
    aloss = np.array([r.get("loss/actor_loss", np.nan) for r in rows], float)
    ks = steps / 1000.0

    plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False})

    # chart_01 — Q-value (critic estimate) vs realized return: the over-estimation story
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ks, q, color=BLUE, lw=1.6, label="Critic Q-value estimate")
    ax.plot(ks, smooth(ret, args.smooth), color=GREEN, lw=1.8,
            label="Realized avg return (smoothed)")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Value")
    ax.set_title(f"Critic Q-value vs. realized return — {suffix}")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_01.png"); plt.close(fig)

    # chart_02 — actor & critic loss
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ks, closs, color=RED, lw=1.5, label="Critic loss (MSBE)")
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Critic loss", color=RED)
    ax.tick_params(axis="y", labelcolor=RED)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(ks, aloss, color=ORANGE, lw=1.5, label="Actor loss  (−Q)")
    ax2.set_ylabel("Actor loss  (−E[Q])", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title(f"Actor & critic loss — {suffix}")
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="upper right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_02.png"); plt.close(fig)

    # chart_03 — episode return & length
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(ks, ret, color=GREEN, lw=0.8, alpha=0.35, label="Avg return (raw)")
    ax.plot(ks, smooth(ret, args.smooth), color=GREEN, lw=2.0, label="Avg return (smoothed)")
    ax.axhline(200, color=BLUE, lw=1.0, ls="--", label="Solved (=200)")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Episode return", color=GREEN)
    ax.tick_params(axis="y", labelcolor=GREEN)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(ks, length, color=ORANGE, lw=1.2, alpha=0.7, label="Episode length")
    ax2.set_ylabel("Episode length", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title(f"Episode return & length — {suffix}")
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="lower right", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_03.png"); plt.close(fig)

    print(f"Charts written to {args.out_dir} | points={len(rows)} | "
          f"peak_return={np.nanmax(ret):.1f} final_return={ret[-1]:.1f} "
          f"peak_Q={np.nanmax(q):.1f} final_Q={q[-1]:.1f}")


if __name__ == "__main__":
    main()
