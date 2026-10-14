"""Generate MAPPO training charts from a local W&B datastore, without network access."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

from MAPPO.utils._common import positive_int

BLUE, RED, GREEN, ORANGE = "#2563eb", "#dc2626", "#16a34a", "#ea580c"
REQUIRED_METRICS = (
    "charts/avg_return",
    "charts/avg_length",
    "stats/value_mean",
    "stats/return_target_mean",
    "loss/policy",
    "loss/value",
    "stats/entropy",
    "stats/approx_kl",
)
OPTIONAL_METRICS = (
    "charts/delivery_rate",
    "charts/mean_progress",
    "development/delivery_rate",
    "stats/skipped_updates",
    "loss/total",
    "stats/explained_variance",
    "charts/survival_rate",
    "charts/learning_rate",
    "progress/agent_transitions",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Explicit local MAPPO training run-<id>.wandb file.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("MAPPO/assets"))
    parser.add_argument("--title", default="MAPPO MultiWalker")
    parser.add_argument("--smooth", type=positive_int, default=9)
    parser.add_argument(
        "--summary",
        type=Path,
        help="Optional raw source series and metric statistics JSON.",
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
            if finite_number(row.get("progress/global_step")):
                rows.append(row)
    finally:
        ds.close()
    rows.sort(key=lambda row: row["progress/global_step"])
    return rows


def series(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    points = [
        (float(row["progress/global_step"]), float(row[metric]))
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


def summarize(rows: list[dict], run_path: Path) -> dict:
    agent_metrics = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("charts/agent_") and key.endswith("_avg_return")
        }
    )
    metrics = {}
    for metric in (*REQUIRED_METRICS, *OPTIONAL_METRICS, *agent_metrics):
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
        "training_rows": len(rows),
        "step_key": "progress/global_step",
        "step_units": "simultaneous parallel environment steps",
        "metrics": metrics,
        "missing_optional_metrics": [
            key for key in OPTIONAL_METRICS if key not in metrics
        ],
        "units": {
            "charts/avg_return": "native undiscounted fixed-original-team mean episode return",
            "charts/avg_length": "simultaneous parallel steps per episode",
            "stats/value_mean": "discounted critic predictions on alive rollout samples",
            "stats/return_target_mean": "discounted GAE-return targets on the same alive rollout samples",
            "stats/entropy": "unsquashed Gaussian entropy proxy for continuous policies",
            "stats/approx_kl": "approximate policy KL diagnostic",
            "charts/survival_rate": "native horizon reached without natural termination; not goal success",
            "progress/agent_transitions": "sum of active-agent counts over parallel steps",
        },
        "interpretation": "Matched value/GAE means are internal training quantities, not independent calibration evidence. Team episodic returns are native and undiscounted. Survival is not success.",
    }


def _save(fig, path: Path) -> None:
    try:
        for axis in fig.axes:
            if axis.get_title():
                axis.set_title(textwrap.fill(axis.get_title(), width=65), fontsize=11)
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
        raise ValueError(
            f"No MAPPO training history with progress/global_step in {args.run}"
        )
    missing = [key for key in REQUIRED_METRICS if not len(series(rows, key)[1])]
    if missing:
        raise ValueError("Missing finite training metrics: " + ", ".join(missing))
    matched = [
        row
        for row in rows
        if all(
            finite_number(row.get(key))
            for key in ("stats/value_mean", "stats/return_target_mean")
        )
    ]
    if not matched:
        raise ValueError("No matched rows containing value and GAE-return target means")
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
    vx, value = series(matched, "stats/value_mean")
    _, targets = series(matched, "stats/return_target_mean")
    top.plot(vx / 1000.0, value, color=BLUE, label="Matched batch critic value mean")
    top.plot(
        vx / 1000.0, targets, color=ORANGE, label="Matched batch GAE-return target mean"
    )
    top.set_ylabel("Discounted per-agent value / target")
    top.set_title(f"Critic predictions and training targets - {args.title}")
    top.legend(loc="best")
    top.text(
        0.02,
        0.02,
        "Matched means are not independent calibration proof",
        transform=top.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    rx, returns = series(rows, "charts/avg_return")
    bottom.plot(
        rx / 1000.0, returns, color=GREEN, alpha=0.35, label="Logged native team return"
    )
    bottom.plot(
        rx / 1000.0,
        smooth(returns, args.smooth),
        color=GREEN,
        label="Team return (smoothed)",
    )
    bottom.set_ylabel("Undiscounted native team episode return")
    bottom.set_xlabel("Parallel environment steps (thousands)")
    bottom.set_title(
        "Native team episode returns have different units from discounted critic targets"
    )
    bottom.legend(loc="best")
    _save(fig, args.out_dir / "chart_01.png")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    px, policy = series(rows, "loss/policy")
    lx, value_loss = series(rows, "loss/value")
    top.plot(px / 1000.0, policy, color=ORANGE, label="Policy loss")
    top.set_ylabel("Policy loss", color=ORANGE)
    top.tick_params(axis="y", labelcolor=ORANGE)
    twin = top.twinx()
    twin.grid(False)
    twin.plot(lx / 1000.0, value_loss, color=RED, label="Value loss")
    twin.set_ylabel("Value loss", color=RED)
    twin.tick_params(axis="y", labelcolor=RED)
    top.set_title(f"Policy/value learning and policy diagnostics - {args.title}")
    _legend(top, twin)
    ex, entropy = series(rows, "stats/entropy")
    kx, kl = series(rows, "stats/approx_kl")
    bottom.plot(
        ex / 1000.0,
        entropy,
        color=GREEN,
        label="Entropy (Gaussian proxy for continuous policy)",
    )
    bottom.set_ylabel("Entropy diagnostic", color=GREEN)
    twin = bottom.twinx()
    twin.grid(False)
    twin.plot(kx / 1000.0, kl, color=BLUE, label="Approximate KL")
    twin.set_ylabel("Approximate KL", color=BLUE)
    bottom.set_xlabel("Parallel environment steps (thousands)")
    _legend(bottom, twin)
    _save(fig, args.out_dir / "chart_02.png")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    top.plot(
        rx / 1000.0, returns, color=GREEN, alpha=0.35, label="Logged native team return"
    )
    top.plot(
        rx / 1000.0,
        smooth(returns, args.smooth),
        color=GREEN,
        label="Team return (smoothed)",
    )
    top.set_ylabel("Undiscounted native team return", color=GREEN)
    twin = top.twinx()
    twin.grid(False)
    sx, lengths = series(rows, "charts/avg_length")
    twin.plot(sx / 1000.0, lengths, color=ORANGE, label="Episode length")
    twin.set_ylabel("Parallel steps per episode", color=ORANGE)
    top.set_title(f"Native team return and episode duration - {args.title}")
    _legend(top, twin)
    agent_metrics = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("charts/agent_") and key.endswith("_avg_return")
        }
    )
    dx, delivery = series(rows, "charts/delivery_rate")
    px, progress = series(rows, "charts/mean_progress")
    hx, horizon_survival = series(rows, "charts/survival_rate")
    if len(delivery):
        bottom.plot(
            dx / 1000.0,
            delivery,
            color=BLUE,
            label="Sampled training: native delivery fraction",
        )
        if len(progress):
            bottom.plot(
                px / 1000.0,
                progress,
                color=ORANGE,
                label="Sampled training: mean course progress",
            )
        bottom.set_ylim(-0.02, 1.02)
        bottom.set_ylabel("Native delivery / course progress")
        bottom.set_title(
            "True endpoint completion, not horizon survival; held-out rates are separate"
        )
    elif len(horizon_survival):
        bottom.plot(
            hx / 1000.0,
            horizon_survival,
            color=BLUE,
            label="Horizon survival (not task success)",
        )
        bottom.set_ylim(-0.02, 1.02)
        bottom.set_ylabel("Completed-episode survival fraction")
        if agent_metrics:
            curves = [series(rows, key) for key in agent_metrics]
            if all(
                np.array_equal(x, curves[0][0]) and np.array_equal(y, curves[0][1])
                for x, y in curves
            ):
                bottom.set_title(
                    "Native shared rewards give identical individual return curves"
                )
    elif agent_metrics:
        for key in agent_metrics:
            ax, values = series(rows, key)
            if len(values):
                bottom.plot(ax / 1000.0, values, label=key.removeprefix("charts/"))
        bottom.set_ylabel("Native individual episode returns")
    else:
        bottom.plot(rx / 1000.0, returns, color=GREEN, label="Native fixed-team return")
        bottom.set_ylabel("Undiscounted native team return")
    bottom.set_xlabel("Parallel environment steps (thousands)")
    bottom.legend(loc="best")
    _save(fig, args.out_dir / "chart_03.png")
    summary = summarize(rows, args.run)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(
        f"Charts written to {args.out_dir}; training rows={len(rows)}; matched rows={len(matched)}"
    )
    return summary


def main(argv=None) -> None:
    make_charts(parse_args(argv))


if __name__ == "__main__":
    main()
