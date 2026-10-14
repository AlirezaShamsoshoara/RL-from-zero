"""Make three honest A3C diagnostics from a local W&B training datastore."""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

from A3C.utils._common import (
    config_metadata,
    positive_int,
    read_config,
    sha256,
    write_json,
)

BLUE, RED, GREEN, ORANGE = "#2563eb", "#dc2626", "#16a34a", "#ea580c"
STEP_KEY = "progress/global_step"
METRICS = (
    "charts/avg_return",
    "charts/avg_length",
    "charts/avg_learning_return",
    "stats/value_mean",
    "stats/return_mean",
    "stats/advantage_mean",
    "loss/policy_loss",
    "loss/value_loss",
    "loss/total_loss",
    "stats/entropy",
)
REQUIRED_METRICS = (
    "charts/avg_return",
    "charts/avg_length",
    "stats/value_mean",
    "stats/return_mean",
    "loss/policy_loss",
    "loss/value_loss",
    "loss/total_loss",
    "stats/entropy",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        type=Path,
        help="Explicit A3C training run-<id>.wandb file; never guessed from the newest run",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("A3C/configs/lunarlander.yaml"),
        help="Exact training YAML for reward/value unit provenance",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("A3C/assets"))
    parser.add_argument("--title", default="A3C LunarLander-v3")
    parser.add_argument(
        "--smooth",
        type=positive_int,
        default=9,
        help="Centered display smoothing over logged points, not environment steps",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Source JSON path; defaults to out-dir/chart_metrics.json",
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=None,
        help="Optional all-episode native evaluate.py JSON for matched initial-state value diagnostics",
    )
    return parser.parse_args(argv)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_history(run_path: str | Path) -> list[dict]:
    """Read flat or repeated nested_key fields locally, without an API client."""
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
            if STEP_KEY not in row:
                if any(metric in row for metric in METRICS):
                    raise ValueError(f"Training metrics lack required {STEP_KEY}")
                continue
            step = row[STEP_KEY]
            if not _is_number(step) or not math.isfinite(step) or step < 0:
                raise ValueError(f"Invalid {STEP_KEY}: {step!r}")
            # Export only known numeric source metrics, never secrets or arbitrary logs.
            clean = {STEP_KEY: step}
            for metric in METRICS:
                if metric not in row:
                    continue
                value = row[metric]
                if not _is_number(value) or not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite or nonnumeric training metric {metric} at step {step}"
                    )
                clean[metric] = value
            rows.append(clean)
    finally:
        ds.close()
    rows.sort(key=lambda row: row[STEP_KEY])
    return rows


def series(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    points = []
    for row in rows:
        if metric not in row:
            continue
        step, value = row.get(STEP_KEY), row[metric]
        if not _is_number(step) or not math.isfinite(step) or step < 0:
            raise ValueError(f"Invalid {STEP_KEY}")
        if not _is_number(value) or not math.isfinite(value):
            raise ValueError(f"Non-finite or nonnumeric {metric}")
        points.append((float(step), float(value)))
    if not points:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    x, y = zip(*points)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("smooth window must be positive")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Cannot smooth non-finite values")
    left, right = (window - 1) // 2, window // 2
    return np.asarray(
        [
            np.mean(values[max(0, i - left) : min(len(values), i + right + 1)])
            for i in range(len(values))
        ],
        dtype=float,
    )


def matched_value_rows(rows: list[dict]) -> list[dict]:
    matched = [
        row for row in rows if "stats/value_mean" in row and "stats/return_mean" in row
    ]
    if not matched:
        raise ValueError(
            "No matched rollout rows containing both stats/value_mean and stats/return_mean"
        )
    for metric in ("stats/value_mean", "stats/return_mean"):
        series(matched, metric)
    return matched


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
        "algorithm": "A3C",
        "source": str(Path(run_path).resolve()),
        "source_sha256": sha256(run_path),
        "training_rows": len(rows),
        "step_key": STEP_KEY,
        "metrics": metrics,
        "missing_metrics": [metric for metric in METRICS if metric not in metrics],
        "source_rows": rows,
        "units": {
            "charts/avg_return": "native undiscounted episode return, logged rolling mean",
            "charts/avg_learning_return": "scaled/shaped training undiscounted episode return, logged rolling mean",
            "charts/avg_length": "environment transitions per episode, logged rolling mean",
            "stats/value_mean": "discounted value prediction mean, scaled/shaped training reward units",
            "stats/return_mean": "matched rollout n-step bootstrapped target mean, scaled/shaped training reward units",
            "loss/policy_loss": "categorical policy gradient loss: -mean(advantage * log_prob)",
            "loss/value_loss": "value head MSE against detached n-step targets",
            "loss/total_loss": "policy + weighted value loss - weighted entropy",
            "stats/entropy": "categorical policy entropy in nats",
        },
        "interpretation": "Value and n-step target means come from matching rollout rows. These are aggregate diagnostics, not independent calibration samples. Native episodic return has different units and is plotted separately. Argmax evaluation changes policy relative to stochastic training.",
    }


def evaluation_pairs(path: Path, config: dict) -> tuple[list[dict], dict]:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("algorithm") != "A3C" or data.get("env_id") != config["env_id"]:
        raise ValueError(
            "Evaluation must be an A3C native evaluation for the same environment"
        )
    if data.get("gamma") != float(config["gamma"]):
        raise ValueError("Evaluation/training gamma mismatch")
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Evaluation has no individual episode records")
    pairs = []
    for episode in episodes:
        if not episode.get("calibration_eligible", False):
            continue
        if (
            not episode.get("terminated")
            or episode.get("truncated")
            or episode.get("capped")
        ):
            raise ValueError("Censored evaluation episode marked calibration eligible")
        value, target = (
            episode.get("initial_native_value"),
            episode.get("native_discounted_return"),
        )
        if (
            not _is_number(value)
            or not _is_number(target)
            or not math.isfinite(value)
            or not math.isfinite(target)
        ):
            raise ValueError(
                "Evaluation contains a non-finite matched value/return pair"
            )
        pairs.append({"seed": episode["seed"], "value": value, "return_target": target})
    if not pairs:
        raise ValueError(
            "No uncensored, unshaped matched native initial-state value/return samples"
        )
    return pairs, {
        "source": str(Path(path).resolve()),
        "source_sha256": sha256(path),
        "checkpoint": data.get("checkpoint"),
        "pairs": pairs,
        "eligible_count": len(pairs),
        "all_episode_count": len(episodes),
        "selection_note": "Only uncensored unshaped episodes can match full native discounted returns. This subset is not an all-episode performance estimate.",
    }


def _title(ax, title: str) -> None:
    ax.set_title(textwrap.fill(title, width=66), pad=12)


def _save(fig, path: Path) -> None:
    try:
        fig.tight_layout(pad=1.7)
        fig.savefig(str(path))
    finally:
        plt.close(fig)


def _draw_series(
    ax, rows: list[dict], metric: str, label: str, color: str, window: int = 1
):
    steps, values = series(rows, metric)
    if not len(values):
        raise ValueError(f"Cannot draw empty {metric}")
    if window > 1:
        ax.plot(
            steps / 1000,
            values,
            color=color,
            alpha=0.3,
            marker=".",
            label=label + " (logged)",
        )
    ax.plot(
        steps / 1000,
        smooth(values, window),
        color=color,
        marker="." if len(values) < 4 else None,
        label=label + (" (smoothed)" if window > 1 else ""),
    )
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.set_xlabel("Environment transitions (thousands)")


def make_charts(args: argparse.Namespace) -> dict:
    config = read_config(args.config)
    rows = load_history(args.run)
    if not rows:
        raise ValueError(f"No A3C training rows with {STEP_KEY} in {args.run}")
    missing = [
        metric for metric in REQUIRED_METRICS if not len(series(rows, metric)[1])
    ]
    if missing:
        raise ValueError(
            "Missing required finite training metrics: " + ", ".join(missing)
        )
    matched = matched_value_rows(rows)
    pairs, eval_source = None, None
    if getattr(args, "evaluation", None) is not None:
        pairs, eval_source = evaluation_pairs(args.evaluation, config)
    summary = summarize(rows, args.run)
    summary["config_provenance"] = config_metadata(args.config, config)
    summary["matched_rollout_rows"] = matched
    summary["evaluation"] = eval_source
    summary["display_smoothing_points"] = args.smooth
    # Validate the entire numeric export before rendering anything.
    json.dumps(summary, allow_nan=False)
    out_dir = Path(args.out_dir)
    summary_path = Path(
        getattr(args, "summary", None) or out_dir / "chart_metrics.json"
    )
    inputs = {Path(args.run).resolve(), Path(args.config).resolve()}
    if getattr(args, "evaluation", None) is not None:
        inputs.add(Path(args.evaluation).resolve())
    png_paths = {
        (out_dir / f"chart_{index:02d}.png").resolve() for index in range(1, 4)
    }
    if summary_path.resolve() in inputs | png_paths or png_paths & inputs:
        raise ValueError("Chart outputs must not overwrite source files or each other")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
        }
    )
    training_units = (
        "Shaped training units"
        if config["reward_shaping"]
        else f"Scaled training units (scale={config['reward_scale']:g})"
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    x, value = series(matched, "stats/value_mean")
    _, target = series(matched, "stats/return_mean")
    axes[0].plot(
        x / 1000,
        value,
        color=BLUE,
        marker=".",
        label="Value head mean on rollout batch",
    )
    axes[0].plot(
        x / 1000, target, color=ORANGE, marker=".", label="Matched n-step target mean"
    )
    axes[0].set_ylabel(training_units)
    axes[0].set_xlabel("Environment transitions (thousands)")
    _title(
        axes[0],
        f"A3C matched rollout means, not independent calibration - {args.title}",
    )
    axes[0].legend(loc="best", fontsize=9)
    if pairs is not None:
        values = [pair["value"] for pair in pairs]
        targets = [pair["return_target"] for pair in pairs]
        axes[1].scatter(
            targets, values, color=BLUE, s=18, label="Matched initial-state samples"
        )
        lo, hi = min(values + targets), max(values + targets)
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        axes[1].plot(
            [lo, hi],
            [lo, hi],
            color="gray",
            linestyle="--",
            label="Equality reference, not data",
        )
        axes[1].set_xlabel("Observed native discounted argmax-policy return")
        axes[1].set_ylabel("Initial V / training reward_scale")
        _title(
            axes[1],
            "A3C uncensored native samples: argmax evaluation differs from stochastic training",
        )
    else:
        axes[1].plot(
            x / 1000,
            value - target,
            color=RED,
            marker=".",
            label="Matched batch-mean residual",
        )
        axes[1].axhline(
            0, color="gray", linestyle="--", label="Zero residual reference"
        )
        axes[1].set_xlabel("Environment transitions (thousands)")
        axes[1].set_ylabel(training_units)
        _title(
            axes[1],
            "A3C value mean minus matched target mean, not sample-level calibration",
        )
    axes[1].legend(loc="best", fontsize=9)
    _save(fig, out_dir / "chart_01.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = (
        ("loss/policy_loss", "Categorical policy loss", ORANGE),
        ("loss/value_loss", "Value head MSE", RED),
        ("loss/total_loss", "Weighted A3C total loss", BLUE),
        ("stats/entropy", "Categorical policy entropy (nats)", GREEN),
    )
    for ax, (metric, label, color) in zip(axes.flat, panels):
        _draw_series(ax, rows, metric, label, color)
        ax.set_ylabel(label)
        _title(ax, label)
    fig.suptitle(
        textwrap.fill(f"A3C shared-backbone learning diagnostics - {args.title}", 80)
    )
    _save(fig, out_dir / "chart_02.png")

    has_learning_return = bool(len(series(rows, "charts/avg_learning_return")[1]))
    fig, axes = plt.subplots(
        3 if has_learning_return else 2,
        1,
        figsize=(10, 10 if has_learning_return else 7),
    )
    _draw_series(
        axes[0], rows, "charts/avg_return", "Native episode return", GREEN, args.smooth
    )
    axes[0].axhline(
        200,
        color=BLUE,
        linestyle="--",
        label="Native return benchmark >= 200, not landing rate",
    )
    axes[0].legend(loc="best", fontsize=9)
    axes[0].set_ylabel("Native undiscounted reward")
    _title(axes[0], f"A3C stochastic-training native episode returns - {args.title}")
    index = 1
    if has_learning_return:
        _draw_series(
            axes[index],
            rows,
            "charts/avg_learning_return",
            "Training learning return",
            BLUE,
            args.smooth,
        )
        axes[index].set_ylabel(training_units + " (undiscounted)")
        _title(
            axes[index],
            "A3C scaled/shaped learning returns: separate units, not native scores",
        )
        index += 1
    _draw_series(
        axes[index], rows, "charts/avg_length", "Episode length", ORANGE, args.smooth
    )
    axes[index].set_ylabel("Transitions per episode")
    _title(axes[index], "A3C logged rolling episode length")
    _save(fig, out_dir / "chart_03.png")
    write_json(
        getattr(args, "summary", None) or out_dir / "chart_metrics.json", summary
    )
    return summary


def main(argv=None) -> None:
    args = parse_args(argv)
    result = make_charts(args)
    print(
        f"Wrote three A3C charts to {args.out_dir}; {result['training_rows']} source rows, {len(result['matched_rollout_rows'])} matched rollout rows"
    )


if __name__ == "__main__":
    main()
