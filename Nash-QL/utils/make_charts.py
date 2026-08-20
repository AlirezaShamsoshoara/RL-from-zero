"""Generate Nash-QL grid-soccer training charts (chart_01/02/03.png) from a wandb run.

Reads the metric history straight from a run's ``.wandb`` datastore file (works
for offline runs too, no network needed) and writes three PNGs:

  - chart_01: self-play outcome rates (agent 0 win / agent 1 win / draw)
  - chart_02: evaluation win rate of the learned agent vs a random opponent
  - chart_03: epsilon (exploration) schedule

Example:
    python Nash-QL/utils/make_charts.py \
        --run wandb/offline-run-XXXX/run-XXXX.wandb \
        --out-dir Nash-QL/assets --title "Nash-QL Grid Soccer"
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

BLUE, RED, GREEN, ORANGE, PURPLE, GRAY = "#2563eb", "#dc2626", "#16a34a", "#ea580c", "#7c3aed", "#6b7280"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Path to run-<id>.wandb")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--smooth", type=int, default=7)
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
    xs, ys = [], []
    for r in rows:
        if metric in r and "progress/episode" in r:
            xs.append(r["progress/episode"])
            ys.append(r[metric])
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys, dtype=float)[order]


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

    plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.unicode_minus": False})

    # chart_01 - self-play outcome composition
    ex, w0 = series(rows, "charts/agent0_win_rate")
    _, w1 = series(rows, "charts/agent1_win_rate")
    _, dr = series(rows, "charts/draw_rate")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(ex):
        ax.plot(ex, smooth(w0, args.smooth), color=BLUE, lw=1.8, label="agent 0 win")
        ax.plot(ex, smooth(w1, args.smooth), color=RED, lw=1.8, label="agent 1 win")
        ax.plot(ex, smooth(dr, args.smooth), color=GRAY, lw=1.8, label="draw")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Rate (self-play)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Self-play outcomes - {suffix}")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_01.png"); plt.close(fig)

    # chart_02 - exploitability curve (headline) with the vs-random win rate as a secondary line
    ex_x, expl_s = series(rows, "exact/exploit_start")
    _, expl_m = series(rows, "exact/exploit_mean")
    vx, vw = series(rows, "eval/win_rate_vs_random")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(ex_x):
        ax.plot(ex_x, expl_s, color=GREEN, lw=2.2, marker="o", ms=4,
                label="exploitability at start (V* - V_vs_BR)")
        ax.plot(ex_x, expl_m, color=ORANGE, lw=1.5, ls="--",
                label="exploitability (state-averaged)")
        ax.axhline(0.0, color=BLUE, lw=1.0, ls=":", label="0 = exact Nash")
        ax.set_ylabel("Exploitability (lower is better)")
    else:
        # Fallback: no exact-eval data in this run, plot vs-random.
        if len(vx):
            ax.plot(vx, vw, color=GREEN, lw=2.0, marker="o", ms=4,
                    label="learned agent 0 win rate")
        ax.set_ylabel("Win rate vs random opponent")
    ax.set_xlabel("Episode")
    ax.set_title(f"Convergence to analytical Nash - {suffix}")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_02.png"); plt.close(fig)

    # chart_02b - head-to-head against the analytical Nash opponent
    hh_x, hh_r = series(rows, "exact/h2h_mean_r0")
    _, vs_x = series(rows, "exact/V_star_start")
    _, hh_w = series(rows, "exact/h2h_win")
    _, hh_l = series(rows, "exact/h2h_loss")
    if len(hh_x):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(hh_x, hh_r, color=GREEN, lw=2.0, marker="o", ms=4,
                label="mean agent-0 return vs exact Nash")
        if len(vs_x):
            ax.plot(hh_x, vs_x, color=BLUE, lw=1.2, ls=":", label="V* at start (game value)")
        ax.plot(hh_x, hh_w, color=ORANGE, lw=1.2, ls="--", label="win rate")
        ax.plot(hh_x, hh_l, color=RED, lw=1.2, ls="--", label="loss rate")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Head-to-head")
        ax.set_title(f"Head-to-head vs analytical Nash opponent - {suffix}")
        ax.legend(loc="best", framealpha=0.9)
        fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_04.png"); plt.close(fig)

    # chart_03 - epsilon schedule
    epx, eps = series(rows, "charts/epsilon")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epx, eps, color=PURPLE, lw=1.8, label="Epsilon (exploration)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title(f"Exploration schedule - {suffix}")
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/chart_03.png"); plt.close(fig)

    peak = np.nanmax(vw) if len(vw) else float("nan")
    min_expl = np.nanmin(expl_s) if len(ex_x) else float("nan")
    print(f"Charts written to {args.out_dir} | rows={len(rows)} | "
          f"peak_eval_win_rate={peak:.3f} | min_exploit_start={min_expl:.4f}")


if __name__ == "__main__":
    main()
