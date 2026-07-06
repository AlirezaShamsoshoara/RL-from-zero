"""Generate IQL training charts (chart_01/02/03.png) from a local wandb run.

Reads the metric history straight from a run's ``.wandb`` datastore file (works
for offline runs too, no network needed) and writes three PNGs:

  - chart_01: evaluation return over training (the offline-RL headline)
  - chart_02: actor / critic / value losses
  - chart_03: AWR diagnostics (mean advantage and policy-extraction weights)

Example:
    python -m IQL.utils.make_charts \
        --run wandb/offline-run-XXXX/run-XXXX.wandb \
        --out-dir IQL/assets --title "IQL Pendulum (mixed)" \
        --behavior -497 --optimal -150
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
    p.add_argument("--behavior", type=float, default=None,
                   help="Optional behavior-policy return reference line.")
    p.add_argument("--optimal", type=float, default=None,
                   help="Optional optimal/near-optimal return reference line.")
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
    return rows


def series(rows: list, metric: str):
    """Return (steps, values) for rows that contain `metric`, x = update step."""
    xs, ys = [], []
    for r in rows:
        if metric in r:
            step = r.get("progress/update", r.get("_step", len(xs)))
            xs.append(step)
            ys.append(r[metric])
    order = np.argsort(xs)
    return np.array(xs)[order] / 1000.0, np.array(ys, dtype=float)[order]


def main() -> None:
    args = parse_args()
    rows = load_history(args.run)
    if not rows:
        raise SystemExit(f"No history rows found in {args.run}")
    os.makedirs(args.out_dir, exist_ok=True)
    suffix = args.title

    plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.unicode_minus": False})

    # chart_01 - evaluation return (the offline-RL headline)
    ex, ey = series(rows, "eval/avg_return")
    _, estd = series(rows, "eval/std_return")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(ex):
        ax.plot(ex, ey, color=GREEN, lw=2.0, marker="o", ms=4, label="Eval return")
        if len(estd) == len(ey):
            ax.fill_between(ex, ey - estd, ey + estd, color=GREEN, alpha=0.15)
    if args.behavior is not None:
        ax.axhline(args.behavior, color=ORANGE, lw=1.2, ls="--", label=f"Behavior policy ({args.behavior:g})")
    if args.optimal is not None:
        ax.axhline(args.optimal, color=BLUE, lw=1.0, ls=":", label=f"Near-optimal ({args.optimal:g})")
    ax.set_xlabel("Gradient updates (thousands)")
    ax.set_ylabel("Episode return")
    ax.set_title(f"Offline eval return - {suffix}")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_01.png"); plt.close(fig)

    # chart_02 - losses
    cx, cl = series(rows, "loss/critic")
    _, vl = series(rows, "loss/value")
    ax_x, al = series(rows, "loss/actor")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cx, cl, color=RED, lw=1.5, label="Critic loss (TD)")
    ax.plot(cx[: len(vl)], vl, color=PURPLE, lw=1.5, label="Value loss (expectile)")
    ax.set_xlabel("Gradient updates (thousands)")
    ax.set_ylabel("Critic / value loss")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(ax_x, al, color=ORANGE, lw=1.5, label="Actor loss (AWR)")
    ax2.set_ylabel("Actor loss", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title(f"Losses - {suffix}")
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_02.png"); plt.close(fig)

    # chart_03 - AWR diagnostics
    adx, adv = series(rows, "stats/mean_advantage")
    wx, wm = series(rows, "stats/weight_mean")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(adx, adv, color=BLUE, lw=1.5, label="Mean advantage (Q - V)")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Gradient updates (thousands)")
    ax.set_ylabel("Mean advantage", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(wx, wm, color=GREEN, lw=1.5, label="AWR weight mean")
    ax2.set_ylabel("exp(adv / temp) weight", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    ax.set_title(f"Advantage-weighted regression diagnostics - {suffix}")
    l1, la1 = ax.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_03.png"); plt.close(fig)

    final = ey[-1] if len(ey) else float("nan")
    best = np.nanmax(ey) if len(ey) else float("nan")
    print(f"Charts written to {args.out_dir} | rows={len(rows)} | "
          f"best_eval={best:.1f} final_eval={final:.1f}")


if __name__ == "__main__":
    main()
