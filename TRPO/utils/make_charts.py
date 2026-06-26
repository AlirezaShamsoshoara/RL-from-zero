"""Generate TRPO training charts (chart_01/02/03.png) from a local wandb run.

Reads the metric history straight from a run's ``.wandb`` datastore file (works
for offline runs too, no network needed) and writes three PNGs:

  - chart_01: episode return (raw + smoothed), with an optional solved line
  - chart_02: policy loss & value loss
  - chart_03: KL divergence & policy entropy (the trust-region diagnostics)

Example:
    python -m TRPO.utils.make_charts \
        --run wandb/offline-run-XXXX/run-XXXX.wandb \
        --out-dir TRPO/assets --title "TRPO BipedalWalker-v3" --max-kl 0.01
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

BLUE, RED, GREEN, ORANGE, PURPLE = "#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Path to run-<id>.wandb")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--smooth", type=int, default=9)
    p.add_argument("--solved", type=float, default=None,
                   help="Optional solved-return reference line.")
    p.add_argument("--max-kl", type=float, default=None,
                   help="Optional max-KL reference line for chart_03.")
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
    rows = [r for r in rows if "progress/global_step" in r]
    rows.sort(key=lambda r: r["progress/global_step"])
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

    steps = np.array([r["progress/global_step"] for r in rows]) / 1000.0
    ret = np.array([r.get("charts/avg_return", np.nan) for r in rows], float)
    length = np.array([r.get("charts/avg_length", np.nan) for r in rows], float)
    ploss = np.array([r.get("loss/policy", np.nan) for r in rows], float)
    vloss = np.array([r.get("loss/value", np.nan) for r in rows], float)
    kl = np.array([r.get("stats/kl", np.nan) for r in rows], float)
    ent = np.array([r.get("stats/entropy", np.nan) for r in rows], float)

    plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.unicode_minus": False})

    # chart_01 - episode return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, ret, color=GREEN, lw=0.8, alpha=0.35, label="Avg return (raw)")
    ax.plot(steps, smooth(ret, args.smooth), color=GREEN, lw=2.0, label="Avg return (smoothed)")
    if args.solved is not None:
        ax.axhline(args.solved, color=BLUE, lw=1.0, ls="--", label=f"Solved ({args.solved:g})")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Episode return")
    ax.set_title(f"Episode return - {suffix}")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_01.png"); plt.close(fig)

    # chart_02 - policy & value loss
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, ploss, color=ORANGE, lw=1.5, label="Policy surrogate loss")
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Policy loss", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=ORANGE)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(steps, vloss, color=RED, lw=1.5, label="Value loss (MSE)")
    ax2.set_ylabel("Value loss", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax.set_title(f"Policy & value loss - {suffix}")
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_02.png"); plt.close(fig)

    # chart_03 - KL & entropy (trust-region diagnostics)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, kl, color=PURPLE, lw=1.5, label="KL(old || new)")
    if args.max_kl is not None:
        ax.axhline(args.max_kl, color=PURPLE, lw=1.0, ls="--", label=f"max_kl ({args.max_kl:g})")
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("KL divergence", color=PURPLE)
    ax.tick_params(axis="y", labelcolor=PURPLE)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(steps, ent, color=GREEN, lw=1.5, label="Policy entropy")
    ax2.set_ylabel("Entropy", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    ax.set_title(f"Trust-region diagnostics (KL & entropy) - {suffix}")
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_03.png"); plt.close(fig)

    print(f"Charts written to {args.out_dir} | points={len(rows)} | "
          f"peak_return={np.nanmax(ret):.1f} final_return={ret[-1]:.1f}")


if __name__ == "__main__":
    main()
