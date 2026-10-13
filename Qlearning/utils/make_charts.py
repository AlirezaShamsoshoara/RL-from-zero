"""Generate tabular Q-learning charts from a local W&B training datastore."""

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

from Qlearning.utils._common import positive_int

BLUE, RED, GREEN, ORANGE = "#2563eb", "#dc2626", "#16a34a", "#ea580c"
METRICS = (
    "charts/avg_return",
    "charts/avg_length",
    "charts/success_rate",
    "charts/epsilon",
    "stats/start_q",
    "stats/discounted_return",
    "stats/td_error",
    "loss/td_mse",
    "progress/steps",
)
REQUIRED_METRICS = METRICS[:-1]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Local Qlearning training run-<id>.wandb file.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("Qlearning/assets"))
    parser.add_argument("--title", default="Q-learning FrozenLake")
    parser.add_argument(
        "--smooth",
        type=positive_int,
        default=9,
        help="Centered smoothing window in logged points, not episodes.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Optional raw source series and statistics in JSON.",
    )
    return parser.parse_args(argv)


def finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
    )


def load_history(run_path: str | Path) -> list[dict]:
    """Read repeated nested keys and retain only episode-indexed training rows."""
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
                try:
                    row[key] = json.loads(item.value_json)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed W&B history JSON for {key!r}") from exc
            if finite_number(row.get("progress/episode")):
                rows.append(row)
    finally:
        ds.close()
    rows.sort(key=lambda row: row["progress/episode"])
    return rows


def series(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    points = [
        (float(row["progress/episode"]), float(row[metric]))
        for row in rows
        if finite_number(row.get(metric))
    ]
    if not points:
        return np.array([], dtype=float), np.array([], dtype=float)
    episodes, values = zip(*points)
    return np.asarray(episodes, dtype=float), np.asarray(values, dtype=float)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
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
        episodes, values = series(rows, metric)
        if len(values):
            metrics[metric] = {
                "count": len(values),
                "first_episode": float(episodes[0]),
                "last_episode": float(episodes[-1]),
                "first": float(values[0]),
                "last": float(values[-1]),
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "episodes": episodes.tolist(),
                "values": values.tolist(),
            }
    return {
        "source": str(run_path.resolve()),
        "training_rows": len(rows),
        "step_key": "progress/episode",
        "metrics": metrics,
        "missing_metrics": [metric for metric in METRICS if metric not in metrics],
        "units": {
            "charts/avg_return": "undiscounted episode reward sum, rolling mean",
            "charts/avg_length": "steps per episode, rolling mean",
            "charts/success_rate": "fraction of FrozenLake episodes with total return > 0",
            "charts/epsilon": "epsilon-greedy exploration probability",
            "stats/start_q": "discounted max-action Q at the initial state",
            "stats/discounted_return": "discounted reward sum under the epsilon-greedy behavior policy",
            "stats/td_error": "mean absolute tabular TD error",
            "loss/td_mse": "mean squared tabular TD error, not a neural-network loss",
            "progress/steps": "cumulative environment transitions",
        },
        "interpretation": (
            "Max start-state Q targets a greedy policy, while observed discounted "
            "training returns use epsilon-greedy behavior. Different policies and "
            "finite episode horizons prevent interpreting their gap as calibration "
            "proof. Raw episode returns are also undiscounted."
        ),
    }


def _save(fig, path: Path) -> None:
    try:
        fig.tight_layout()
        fig.savefig(str(path))
    finally:
        plt.close(fig)


def _legend(ax, twin) -> None:
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")


def make_charts(args: argparse.Namespace) -> dict:
    rows = load_history(args.run)
    if not rows:
        raise ValueError(f"No training history with progress/episode in {args.run}")
    missing = [
        metric for metric in REQUIRED_METRICS if not len(series(rows, metric)[1])
    ]
    if missing:
        raise ValueError("Missing finite training metrics: " + ", ".join(missing))
    matched = [
        row
        for row in rows
        if all(
            finite_number(row.get(metric))
            for metric in ("stats/start_q", "stats/discounted_return")
        )
    ]
    if not matched:
        raise ValueError(
            "No matched rows containing start Q and discounted behavior return"
        )
    _, success = series(rows, "charts/success_rate")
    _, epsilon = series(rows, "charts/epsilon")
    if np.any((success < 0) | (success > 1)) or np.any((epsilon < 0) | (epsilon > 1)):
        raise ValueError("Success rate and epsilon must be fractions between 0 and 1")
    for metric in ("stats/td_error", "loss/td_mse"):
        if np.any(series(rows, metric)[1] < 0):
            raise ValueError(f"{metric} must be nonnegative")
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

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    qx, start_q = series(matched, "stats/start_q")
    _, discounted = series(matched, "stats/discounted_return")
    top.plot(qx, start_q, color=BLUE, label="Max start-state Q")
    top.plot(qx, discounted, color=ORANGE, label="Discounted epsilon-behavior return")
    top.set_ylabel("Discounted Q / reward sum")
    top.set_title(f"Start Q and behavior returns - {args.title}")
    top.legend(loc="best")
    top.text(
        0.02,
        0.02,
        "Greedy Q vs epsilon behavior: not calibration proof",
        transform=top.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    rx, returns = series(rows, "charts/avg_return")
    bottom.plot(rx, returns, color=GREEN, alpha=0.35, label="Logged episode return")
    bottom.plot(
        rx, smooth(returns, args.smooth), color=GREEN, label="Episode return (smoothed)"
    )
    bottom.set_ylabel("Undiscounted episode return")
    bottom.set_xlabel("Training episode")
    bottom.set_title("Raw episode reward sums are not discounted Q targets")
    bottom.legend(loc="best")
    _save(fig, args.out_dir / "chart_01.png")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    mx, mse = series(rows, "loss/td_mse")
    tx, abs_td = series(rows, "stats/td_error")
    top.plot(mx, mse, color=RED, label="Mean squared TD error")
    top.set_ylabel("TD-MSE", color=RED)
    top.tick_params(axis="y", labelcolor=RED)
    twin = top.twinx()
    twin.grid(False)
    twin.plot(tx, abs_td, color=BLUE, label="Mean absolute TD error")
    twin.set_ylabel("Mean absolute TD error", color=BLUE)
    twin.tick_params(axis="y", labelcolor=BLUE)
    top.set_title(f"Tabular update diagnostics - {args.title}")
    _legend(top, twin)
    ex, epsilon = series(rows, "charts/epsilon")
    bottom.plot(ex, epsilon, color=ORANGE, label="Epsilon")
    bottom.set_ylim(-0.02, 1.02)
    bottom.set_ylabel("Exploration probability")
    bottom.set_xlabel("Training episode")
    bottom.legend(loc="best")
    _save(fig, args.out_dir / "chart_02.png")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    top.plot(rx, returns, color=GREEN, alpha=0.35, label="Logged episode return")
    top.plot(
        rx, smooth(returns, args.smooth), color=GREEN, label="Episode return (smoothed)"
    )
    top.set_ylabel("Undiscounted episode return", color=GREEN)
    top.tick_params(axis="y", labelcolor=GREEN)
    twin = top.twinx()
    twin.grid(False)
    lx, lengths = series(rows, "charts/avg_length")
    twin.plot(lx, lengths, color=ORANGE, label="Episode length")
    twin.set_ylabel("Steps per episode", color=ORANGE)
    twin.tick_params(axis="y", labelcolor=ORANGE)
    top.set_title(f"Return, length, and FrozenLake success - {args.title}")
    _legend(top, twin)
    sx, success = series(rows, "charts/success_rate")
    bottom.plot(sx, success, color=BLUE, label="Success rate (return > 0)")
    bottom.set_ylim(-0.02, 1.02)
    bottom.set_ylabel("Success fraction")
    bottom.set_xlabel("Training episode")
    bottom.legend(loc="best")
    _save(fig, args.out_dir / "chart_03.png")

    summary = summarize(rows, args.run)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(
        f"Charts written to {args.out_dir}; training rows={len(rows)}; "
        f"matched Q/behavior rows={len(matched)}"
    )
    return summary


def main(argv=None) -> None:
    make_charts(parse_args(argv))


if __name__ == "__main__":
    main()
