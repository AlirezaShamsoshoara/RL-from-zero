"""Generate native-reward DQN charts from a local W&B training datastore."""

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

from deepQN.utils._common import positive_int

BLUE, RED, GREEN, ORANGE = "#2563eb", "#dc2626", "#16a34a", "#ea580c"
METRICS = (
    "charts/avg_return",
    "charts/avg_length",
    "charts/epsilon",
    "charts/exploratory_step_fraction",
    "charts/buffer_size",
    "charts/termination_rate",
    "charts/success_rate",
    "stats/start_q",
    "stats/discounted_return",
    "stats/q_value",
    "stats/target_value",
    "stats/td_error",
    "loss/q_loss",
)
REQUIRED_METRICS = (
    "charts/avg_return",
    "charts/avg_length",
    "charts/epsilon",
    "charts/termination_rate",
    "stats/q_value",
    "stats/target_value",
    "stats/td_error",
    "loss/q_loss",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Explicit local DQN training run-<id>.wandb file.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("deepQN/assets"))
    parser.add_argument("--title", default="DQN MountainCar-v0")
    parser.add_argument(
        "--env-id",
        default="MountainCar-v0",
        help="Training environment ID; goal success is specific to MountainCar-v0.",
    )
    parser.add_argument(
        "--smooth",
        type=positive_int,
        default=9,
        help="Centered display smoothing in logged points, not environment steps.",
    )
    parser.add_argument(
        "--summary", type=Path, help="Optional raw metric series/statistics JSON."
    )
    return parser.parse_args(argv)


def finite_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(value)
    )


def load_history(run_path: str | Path) -> list[dict]:
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
            if finite_number(row.get("progress/step")):
                rows.append(row)
    finally:
        ds.close()
    rows.sort(key=lambda row: row["progress/step"])
    return rows


def series(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    points = [
        (float(row["progress/step"]), float(row[metric]))
        for row in rows
        if finite_number(row.get(metric))
    ]
    if not points:
        return np.array([], dtype=float), np.array([], dtype=float)
    steps, values = zip(*points)
    return np.asarray(steps, dtype=float), np.asarray(values, dtype=float)


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


def summarize(rows: list[dict], run_path: Path, env_id: str) -> dict:
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
                "steps": steps.tolist(),
                "values": values.tolist(),
            }
    return {
        "source": str(run_path.resolve()),
        "env_id": env_id,
        "training_rows": len(rows),
        "step_key": "progress/step",
        "metrics": metrics,
        "missing_metrics": [metric for metric in METRICS if metric not in metrics],
        "units": {
            "charts/avg_return": "native undiscounted epsilon-behavior episode reward sum, rolling mean",
            "charts/avg_length": "steps per episode, rolling mean",
            "charts/epsilon": "epsilon-greedy exploration probability",
            "charts/buffer_size": "replay transition count",
            "charts/termination_rate": "fraction of episodes ending in true termination, not general success",
            "charts/success_rate": "MountainCar-v0 only: goal reached by true termination",
            "stats/start_q": "discounted maximum online Q at episode start",
            "stats/discounted_return": "discounted native reward sum under epsilon-greedy behavior",
            "stats/q_value": "mean selected-action online Q on the replay batch",
            "stats/target_value": "mean bootstrapped TD target on the same replay batch",
            "stats/td_error": "mean absolute replay TD error",
            "loss/q_loss": "mean squared replay TD error",
        },
        "interpretation": (
            "Replay Q and bootstrapped TD targets are matched training quantities, "
            "not independent realized episode returns or proof of calibration. "
            "Start-state max Q targets a greedy policy; discounted training returns "
            "use epsilon behavior and capped horizons. Raw returns are undiscounted "
            "native rewards. Negative MountainCar return is not a failure criterion."
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
    ax.legend(
        lines + lines2,
        labels + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        fontsize=9,
    )


def make_charts(args: argparse.Namespace) -> dict:
    rows = load_history(args.run)
    if not rows:
        raise ValueError(f"No DQN training history with progress/step in {args.run}")
    mountaincar = args.env_id == "MountainCar-v0"
    required = REQUIRED_METRICS + (("charts/success_rate",) if mountaincar else ())
    missing = [metric for metric in required if not len(series(rows, metric)[1])]
    if missing:
        raise ValueError("Missing finite training metrics: " + ", ".join(missing))
    matched = [
        row
        for row in rows
        if all(
            finite_number(row.get(metric))
            for metric in ("stats/q_value", "stats/target_value")
        )
    ]
    if not matched:
        raise ValueError("No matched replay rows containing both Q and TD target means")
    for metric in ("charts/epsilon", "charts/termination_rate", "charts/success_rate"):
        _, values = series(rows, metric)
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f"{metric} must be a fraction between 0 and 1")
    for metric in ("loss/q_loss", "stats/td_error"):
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
    qx, q = series(matched, "stats/q_value")
    _, targets = series(matched, "stats/target_value")
    top.plot(qx / 1000.0, q, color=BLUE, label="Replay selected-action Q mean")
    top.plot(
        qx / 1000.0,
        targets,
        color=ORANGE,
        linestyle="--",
        label="Matched bootstrapped TD target mean",
    )
    top.set_ylabel("Discounted replay Q / TD target")
    top.set_title(f"Replay Q and training targets - {args.title}")
    top.legend(loc="best")
    top.text(
        0.02,
        0.02,
        "Bootstrapped targets are not independent calibration evidence",
        transform=top.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    rx, returns = series(rows, "charts/avg_return")
    bottom.plot(
        rx / 1000.0, returns, color=GREEN, alpha=0.35, label="Logged native return"
    )
    bottom.plot(
        rx / 1000.0,
        smooth(returns, args.smooth),
        color=GREEN,
        label="Native return (smoothed)",
    )
    bottom.set_ylabel("Undiscounted native episode return")
    bottom.set_xlabel("Environment transitions (thousands)")
    bottom.set_title("Episode reward sums are separate from discounted replay targets")
    bottom.legend(loc="best")
    _save(fig, args.out_dir / "chart_01.png")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    mx, mse = series(rows, "loss/q_loss")
    tx, abs_td = series(rows, "stats/td_error")
    top.plot(mx / 1000.0, mse, color=RED, label="Q loss (TD-MSE)")
    top.set_ylabel("Mean squared TD error", color=RED)
    top.tick_params(axis="y", labelcolor=RED)
    twin = top.twinx()
    twin.grid(False)
    twin.plot(tx / 1000.0, abs_td, color=BLUE, label="Mean absolute TD error")
    twin.set_ylabel("Mean absolute TD error", color=BLUE)
    twin.tick_params(axis="y", labelcolor=BLUE)
    top.set_title(f"Q-network learning diagnostics - {args.title}")
    _legend(top, twin)
    ex, epsilon = series(rows, "charts/epsilon")
    bottom.plot(ex / 1000.0, epsilon, color=ORANGE, label="Nominal epsilon")
    ax_steps, observed = series(rows, "charts/exploratory_step_fraction")
    if len(observed):
        bottom.plot(
            ax_steps / 1000.0,
            observed,
            color=BLUE,
            label="Observed exploratory-step fraction",
        )
    bottom.set_ylim(-0.02, 1.02)
    bottom.set_ylabel("Exploration step fraction")
    bottom.set_xlabel("Environment transitions (thousands)")
    bottom.legend(loc="best")
    _save(fig, args.out_dir / "chart_02.png")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    top.plot(
        rx / 1000.0, returns, color=GREEN, alpha=0.35, label="Logged native return"
    )
    top.plot(
        rx / 1000.0,
        smooth(returns, args.smooth),
        color=GREEN,
        label="Native return (smoothed)",
    )
    top.set_ylabel("Undiscounted native return", color=GREEN)
    top.tick_params(axis="y", labelcolor=GREEN)
    twin = top.twinx()
    twin.grid(False)
    lx, lengths = series(rows, "charts/avg_length")
    twin.plot(lx / 1000.0, lengths, color=ORANGE, label="Episode length")
    twin.set_ylabel("Steps per episode", color=ORANGE)
    twin.tick_params(axis="y", labelcolor=ORANGE)
    top.set_title(f"Native return and length - {args.title}")
    _legend(top, twin)
    metric = "charts/success_rate" if mountaincar else "charts/termination_rate"
    sx, rate = series(rows, metric)
    label = (
        "MountainCar goal success (true termination)"
        if mountaincar
        else "Termination rate (not success)"
    )
    bottom.plot(sx / 1000.0, rate, color=BLUE, label=label)
    bottom.set_ylim(-0.02, 1.02)
    bottom.set_ylabel(
        "Goal success fraction" if mountaincar else "Termination fraction"
    )
    bottom.set_xlabel("Environment transitions (thousands)")
    bottom.legend(loc="best")
    _save(fig, args.out_dir / "chart_03.png")

    summary = summarize(rows, args.run, args.env_id)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(
        f"Charts written to {args.out_dir}; training rows={len(rows)}; matched replay rows={len(matched)}"
    )
    return summary


def main(argv=None) -> None:
    make_charts(parse_args(argv))


if __name__ == "__main__":
    main()
