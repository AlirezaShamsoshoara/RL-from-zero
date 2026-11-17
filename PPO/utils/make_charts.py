"""Generate PPO training charts from a local W&B datastore without network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

from PPO.utils._common import positive_int

BLUE, RED, GREEN, ORANGE = "#2563eb", "#dc2626", "#16a34a", "#ea580c"
METRICS = (
    "charts/avg_return",
    "charts/avg_length",
    "stats/value_mean",
    "stats/return_target_mean",
    "stats/explained_variance",
    "loss/policy",
    "loss/value",
    "loss/total",
    "stats/entropy",
    "stats/approx_kl",
)
REQUIRED_METRICS = (
    "charts/avg_return",
    "charts/avg_length",
    "stats/value_mean",
    "stats/return_target_mean",
    "loss/policy",
    "loss/value",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", required=True, type=Path, help="Local wandb/<run>/run-<id>.wandb file."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("PPO/assets"))
    parser.add_argument("--title", default="PPO")
    parser.add_argument(
        "--smooth",
        type=positive_int,
        default=9,
        help="Centered smoothing window in logged points, not env steps.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Optional JSON summary of raw source metrics and their units.",
    )
    return parser.parse_args(argv)


def load_history(run_path: str | Path) -> list[dict]:
    """Read history records; nested_key is a repeated field, not a scalar key."""
    ds = datastore.DataStore()
    ds.open_for_scan(str(run_path))
    rows = []
    try:
        while True:
            data = ds.scan_data()
            if data is None:
                break
            record = pb.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") != "history":
                continue
            row = {}
            for item in record.history.item:
                nested = list(item.nested_key)
                key = "/".join(nested) if nested else item.key
                row[key] = json.loads(item.value_json)
            step = row.get("progress/global_step")
            if isinstance(step, (int, float)) and np.isfinite(step):
                rows.append(row)
    finally:
        ds.close()
    rows.sort(key=lambda row: row["progress/global_step"])
    return rows


def series(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    points = [
        (float(row["progress/global_step"]), float(row[metric]))
        for row in rows
        if isinstance(row.get(metric), (int, float)) and np.isfinite(row[metric])
    ]
    if not points:
        return np.array([], dtype=float), np.array([], dtype=float)
    steps, values = zip(*points)
    return np.asarray(steps, dtype=float), np.asarray(values, dtype=float)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving mean for display; source summaries remain unsmoothed."""
    return np.asarray(
        [
            np.mean(
                values[max(0, i - window // 2) : min(len(values), i + window // 2 + 1)]
            )
            for i in range(len(values))
        ],
        dtype=float,
    )


def summarize(rows: list[dict], run_path: Path) -> dict:
    metrics = {}
    for metric in METRICS:
        steps, values = series(rows, metric)
        if len(values):
            metrics[metric] = {
                "count": len(values),
                "first_step": float(steps[0]),
                "last_step": float(steps[-1]),
                "first": float(values[0]),
                "last": float(values[-1]),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return {
        "source": str(run_path.resolve()),
        "training_rows": len(rows),
        "step_key": "progress/global_step",
        "metrics": metrics,
        "missing_metrics": [metric for metric in METRICS if metric not in metrics],
        "units": {
            "charts/avg_return": "undiscounted episode reward sum, rolling mean",
            "charts/avg_length": "environment steps per episode, rolling mean",
            "stats/value_mean": "discounted value estimate on the rollout batch",
            "stats/return_target_mean": "discounted GAE-return target on the same rollout batch",
        },
        "interpretation": (
            "Value estimates and GAE targets share a batch and discounted units. "
            "Their means alone do not establish calibration; episode returns are "
            "undiscounted and must not be treated as matching value targets."
        ),
    }


def _save(fig, path: Path) -> None:
    try:
        fig.tight_layout()
        fig.savefig(str(path))
    finally:
        plt.close(fig)


def make_charts(args: argparse.Namespace) -> dict:
    rows = load_history(args.run)
    if not rows:
        raise ValueError(f"No training history with progress/global_step in {args.run}")
    missing = [
        metric for metric in REQUIRED_METRICS if not len(series(rows, metric)[1])
    ]
    if missing:
        raise ValueError("Missing finite training metrics: " + ", ".join(missing))
    matched = [
        row
        for row in rows
        if all(
            isinstance(row.get(metric), (int, float)) and np.isfinite(row[metric])
            for metric in ("stats/value_mean", "stats/return_target_mean")
        )
    ]
    if not matched:
        raise ValueError(
            "No matched rows containing both value and GAE-return target means"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )

    # Compare like-for-like discounted batch quantities, not episodic reward sums.
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    vx, value = series(matched, "stats/value_mean")
    _, target = series(matched, "stats/return_target_mean")
    top.plot(vx / 1000.0, value, color=BLUE, label="Rollout value mean")
    top.plot(vx / 1000.0, target, color=ORANGE, label="Matched GAE-return target mean")
    top.set_ylabel("Discounted batch value / target")
    top.set_title(f"Value and matched rollout targets - {args.title}")
    top.legend(loc="best")
    rx, ret = series(rows, "charts/avg_return")
    bottom.plot(
        rx / 1000.0, ret, color=GREEN, alpha=0.35, label="Logged episode return"
    )
    bottom.plot(
        rx / 1000.0,
        smooth(ret, args.smooth),
        color=GREEN,
        label="Episode return (smoothed)",
    )
    bottom.set_ylabel("Undiscounted episode return")
    bottom.set_xlabel("Environment transitions (thousands)")
    bottom.set_title(
        "Episode returns have different units from discounted value targets"
    )
    bottom.legend(loc="best")
    _save(fig, args.out_dir / "chart_01.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    px, policy_loss = series(rows, "loss/policy")
    lx, value_loss = series(rows, "loss/value")
    ax.plot(px / 1000.0, policy_loss, color=ORANGE, label="Policy loss")
    ax.set_xlabel("Environment transitions (thousands)")
    ax.set_ylabel("Policy loss", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=ORANGE)
    twin = ax.twinx()
    twin.grid(False)
    twin.plot(lx / 1000.0, value_loss, color=RED, label="Value loss")
    twin.set_ylabel("Value loss", color=RED)
    twin.tick_params(axis="y", labelcolor=RED)
    ax.set_title(f"Policy and value loss - {args.title}")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    _save(fig, args.out_dir / "chart_02.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(rx / 1000.0, ret, color=GREEN, alpha=0.35, label="Logged episode return")
    ax.plot(
        rx / 1000.0,
        smooth(ret, args.smooth),
        color=GREEN,
        label="Episode return (smoothed)",
    )
    ax.set_xlabel("Environment transitions (thousands)")
    ax.set_ylabel("Undiscounted episode return", color=GREEN)
    ax.tick_params(axis="y", labelcolor=GREEN)
    twin = ax.twinx()
    twin.grid(False)
    ex, length = series(rows, "charts/avg_length")
    twin.plot(ex / 1000.0, length, color=ORANGE, label="Episode length")
    twin.set_ylabel("Steps per episode", color=ORANGE)
    twin.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title(f"Episode return and length - {args.title}")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    _save(fig, args.out_dir / "chart_03.png")

    summary = summarize(rows, args.run)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(
        f"Charts written to {args.out_dir}; training rows={len(rows)}; "
        f"matched value/target rows={len(matched)}"
    )
    return summary


def main(argv=None) -> None:
    make_charts(parse_args(argv))


if __name__ == "__main__":
    main()
